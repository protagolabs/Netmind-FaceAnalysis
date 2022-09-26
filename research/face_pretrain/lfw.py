#!/usr/bin/env python
# encoding: utf-8
'''
@author: xingdi
'''

import numpy as np
from PIL import Image
import os
import torch.utils.data as data

import torch
import mat73

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class LFW(data.Dataset):
    def __init__(self, root, file_list, split="train", transform=None, target_transform=None, loader=default_loader):

        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if split=="train":
            self.idx = mat73.loadmat(os.path.join(self.root, "indices_train_test.mat"))['indices_img_train']
        else:
            self.idx = mat73.loadmat(os.path.join(self.root, "indices_train_test.mat"))['indices_img_test']
            
        df = mat73.loadmat(os.path.join(self.root, file_list))

        name_all = df['name']
        target_all = df['label']


        self.names = [name_all[int(i-1)].replace("\\", "/") for i in self.idx]

        self.targets = [target_all[int(i-1)] for i in self.idx]


    def __getitem__(self, index):


        sample = self.loader(os.path.join(self.root, 'lfw/' + self.names[index]))
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.names)
