from __future__ import print_function
import torch.utils.data as data

from os import listdir
from PIL import Image
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_data, image_dir_target, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.data_filenames = []
        self.target_filenames = []
        count = 1
        for x in listdir(image_dir_data):
            filename = image_dir_data + '/Canon_' + str(count) + '_LR2.png'
            self.data_filenames.append(filename)
            count += 1
        count = 1
        for x in listdir(image_dir_target):
            filename = image_dir_target + '/Canon_' + str(count) + '_HR.png'
            self.target_filenames.append(filename)
            count += 1

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        target = load_img(self.target_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.data_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = "Canon"
    train_dir = join(root_dir, "Train/data")
    target_train = join(root_dir, "Train/target")
    crop_size = calculate_valid_crop_size(180, upscale_factor)

    return DatasetFromFolder(train_dir, target_train,
                             input_transform=input_transform(crop_size / 2, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = "Canon"
    test_dir = join(root_dir, "Test/data")
    target_test = join(root_dir, "Test/target")
    crop_size = calculate_valid_crop_size(180, upscale_factor)

    return DatasetFromFolder(test_dir, target_test,
                             input_transform=input_transform(crop_size / 2, upscale_factor),
                             target_transform=target_transform(crop_size))
