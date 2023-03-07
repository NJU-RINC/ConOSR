# Some helper functions for PyTorch, including:

import os
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from load_data_online import CostumeImageFolder, CostumeMixedImageFolder
import shutil
import numpy as np
import random
from skimage import io
from RandAugment import RandAugment
import scipy.spatial.distance
from PIL import ImageFilter

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_MNIST(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform
    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_MNIST_contrastive(roots, category_indexs, batchSize, shuffle=True):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_SVHN(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform
    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_SVHN_contrastive(roots, category_indexs, batchSize, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes


def load_cifar(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes


def load_cifar_contrastive(roots, category_indexs, batchSize, shuffle=True):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes



def load_ImageNet200(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_ImageNet200_contrastive(roots, category_indexs, batchSize, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=4)
    return dataLoader, data_classes

def load_ImageNet_resize(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform


    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes

def load_ImageNet_crop(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if train:
        transform = train_transform
    else:
        transform = test_transform

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def get_onehot_labels(labels, classid_list):
    targets = torch.zeros([labels.shape[0], len(classid_list)], requires_grad=False).to(labels.device)
    for j, label in enumerate(labels):
        if label.item() not in classid_list:
            continue
        index = classid_list.index(label.item())
        targets[j, index] = 1
    return targets

def get_smooth_labels(labels, classid_list, smoothing_coeff=0.1):
    label_positive = 1 - smoothing_coeff
    label_negative = smoothing_coeff / (len(classid_list)-1)

    targets = label_negative * torch.ones([labels.shape[0], len(classid_list)], requires_grad=False).to(labels.device)
    for j, label in enumerate(labels):
        if label.item() not in classid_list:
            continue
        index = classid_list.index(label.item())
        targets[j, index] = label_positive
    return targets