from __future__ import print_function
import numpy as np
from torch import autograd
import torchvision.models as models
import torch
import torch.nn as nn
from torchsummary import summary
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys

from prepare_dataset import prepare_dataset
from arch_model import arch_model

upscale_factor = 2
batchSize = 2
testBatchSize = 2
nEpochs = 200
lr = 0.0002
threads = 4
seed = 123

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed)

print('===> Loading datasets')

train_set = prepare_dataset.get_training_set(upscale_factor)
test_set = prepare_dataset.get_test_set(upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)

print('===> Building model')
model_G = arch_model.MSRResNet().to(device)
summary(model_G, input_size=(3, 150, 150)) # 90 90

model_D = arch_model.Discriminator_VGG_128(3, 64).to(device)
summary(model_D, input_size=(3, 300, 300)) # 180 180

vgg_model = models.vgg19(pretrained=True).to(device)
feature_extractor = arch_model.LossNetwork(vgg_model)
feature_extractor.eval()

### Loss function ###

loss_G = nn.MSELoss().to(device)
loss_D = nn.BCELoss().to(device)

loss_mse = nn.MSELoss().to(device)
loss_l1 = nn.L1Loss().to(device)

### Optimzers ###

optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(0.9, 0.999))
optimizer_D = optim.SGD(model_D.parameters(), lr=lr / 100, momentum=0.9, nesterov=True)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def pretrain_G(epoch):
    epoch_loss_gen = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer_G.zero_grad()
        result = model_G(input)
        loss = loss_G(result, target)

        loss.backward()
        optimizer_G.step()

        epoch_loss_gen += loss.item()

    print("===> Epoch {} Complete: Avg. Loss_Generator: {:.4f}".format(epoch, epoch_loss_gen / len(training_data_loader)))


def pretrain_D(epoch):
    epoch_loss_gen = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        valid = Tensor(np.ones((input.size(0), *[1]))).to(device)
        fake = Tensor(np.zeros((input.size(0), *[1]))).to(device)

        optimizer_D.zero_grad()

        loss_real = loss_D(model_D(target), valid)
        loss_fake = loss_D(model_D(model_G(input)), fake)

        total_loss_D = loss_real + loss_fake

        total_loss_D.backward()
        optimizer_D.step()

        epoch_loss_gen += total_loss_D.item()

    print("===> Epoch {} Complete: Avg. Loss_Generator: {:.4f}".format(epoch, epoch_loss_gen / len(training_data_loader)))


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#############
#   Train   #
#############
writer = SummaryWriter()
lambda_gp = 10

def train(epoch):
    epoch_loss_D, epoch_loss_G = 0, 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        valid = torch.ones(input.size(0), *[1]).to(device)
        fake = torch.zeros(input.size(0), *[1]).to(device)

        ### Train Discriminator ###

        optimizer_D.zero_grad()

        loss_real = loss_D(model_D(target), valid)
        loss_fake = loss_D(model_D(model_G(input)), fake)

        gp = compute_gradient_penalty(model_D, target.data, model_G(input).data)

        total_loss_D = loss_real + loss_fake# + lambda_gp * gp

        total_loss_D.backward()
        optimizer_D.step()

        ### Train Generator ###
        optimizer_G.zero_grad()

        result_G = model_G(input)
        gan_loss = loss_D(model_D(result_G), valid)

        gen_features = feature_extractor(result_G)
        real_features = feature_extractor(target)
        loss_content = loss_G(gen_features, real_features)
        mse = loss_G(result_G, target)

        total_loss_G = gan_loss * 1e-3 + mse + loss_content * 1e-3

        total_loss_G.backward()
        optimizer_G.step()

        epoch_loss_D += total_loss_D.item()
        epoch_loss_G += total_loss_G.item()

        ### Log ###
        print("===> Epoch[{}]({}/{}): Loss_Discriminator: {:.4f}".format(epoch, iteration, len(training_data_loader),
                                                                  total_loss_D.item()))
        print("===> Epoch[{}]({}/{}): Loss_Generator: {:.4f}".format(epoch, iteration, len(training_data_loader), total_loss_G.item()))

    ### Total LOG ###
    print("===> Epoch {} Complete: Avg. Loss_Discriminator: {:.4f}".format(epoch, epoch_loss_D / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss_Generator: {:.4f}".format(epoch, epoch_loss_G / len(training_data_loader)))

    writer.add_scalar('SRGAN_Loss/Discriminator', epoch_loss_D / len(training_data_loader), epoch)
    writer.add_scalar('SRGAN_Loss/Generator', epoch_loss_G / len(training_data_loader), epoch)

    return epoch_loss_D / len(training_data_loader), epoch_loss_G / len(training_data_loader)


def test(epoch):
    avg_psnr = 0
    total_loss = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model_G(input)
            mse = loss_mse(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            total_loss += mse.item()
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. Loss: {:.4f} dB".format(total_loss / len(testing_data_loader)))


def train_test():
    best_loss_D, best_loss_G = sys.float_info.max, sys.float_info.max
    list_loss_G = []
    list_loss_D = []
    for epoch in range(1, nEpochs + 1):
        #model_G.load_state_dict(torch.load("Pretrain_GEN"))
        model_G.train()
        #model_D.load_state_dict(torch.load("Pretrain_DISC"))
        model_D.train()
        loss_per_epoch_D, loss_per_epoch_G = train(epoch)
        list_loss_G.append(loss_per_epoch_G)
        list_loss_D.append(loss_per_epoch_D)
        if loss_per_epoch_D < best_loss_D and loss_per_epoch_G < best_loss_G:
            best_loss_D = loss_per_epoch_D
            best_loss_G = loss_per_epoch_G
            model_out_path_D = "models/Discriminator.pth"
            model_out_path_G = "models/Generator.pth"
            torch.save(model_D.state_dict(), model_out_path_D)
            torch.save(model_G.state_dict(), model_out_path_G)
            print("Checkpoint saved to {} and {} #########################################################".format(model_out_path_D, model_out_path_G))
        test(epoch)

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ep = np.arange(1, nEpochs + 1, 1)
    ax1.plot(ep, list_loss_G)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train_Generator')
    fig1.tight_layout()
    ax1.grid()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    ep = np.arange(1, nEpochs + 1, 1)
    ax2.plot(ep, list_loss_D)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Train_Discriminator')
    fig2.tight_layout()
    ax2.grid()
    plt.show()
