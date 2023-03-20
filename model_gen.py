'''
Name: Agosh Saini
Contact: Agosh.Saini@gmail.com
Website: agoshsaini.com
'''


# --- Imports --- #
import torch
import torchvision

# --- path to dataset --- #
path = 'jpeg-192x192'

# --- randomly transform images --- #
train_data_transform = torchvision.transform.compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(150),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

test_data_transform = torchvision.transform.compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(150),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

# --- creating splits for training and testing --- #
train_data = torchvision.datasets.ImageFolder(path + '/train', transform=train_data_transform)
test_data = torchvision.datasets.ImageFolder(path + '/train', transform=test_data_transform)

# --- parameters for the model --- #

