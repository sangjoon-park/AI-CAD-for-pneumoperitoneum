import logging
import numpy as np
import os, os.path
from PIL import Image
from torch.utils import data
import torch
import glob
import random
import os
import cv2
import utils

import torch
from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import matplotlib.pyplot as plt

from jf_utils.imgaug import GetTransforms
from jf_utils.utils import transform

from PIL import ImageFilter, ImageOps

import main_dino

class CXR_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, view, data_dir, size=1.0, transforms=None, mode='train', labeled=False):
        'Initialization'
        self.dim = (256, 256)
        self.n_classes = 1
        self.transforms = transforms
        self.mode = mode
        self.labeled = labeled
        self.view = view

        self.total_images = {}

        # PNG and JPG image lists
        if self.mode == 'train':
            if self.labeled == True:
                png_lists = glob.glob(data_dir + 'labeled/' + '**/*.png', recursive=True)
                jpg_lists = glob.glob(data_dir + 'labeled/' + '**/*.jpg', recursive=True)
                for png in png_lists:
                    if self.view == 'erect':
                        if 'erect' in png:
                            self.total_images[png] = 'label'
                        else:
                            pass
                    elif self.view == 'supine':
                        if 'supine' in png:
                            self.total_images[png] = 'label'
                        else:
                            pass
                    elif self.view == 'all':
                        self.total_images[png] = 'label'

                for jpg in jpg_lists:
                    if self.view == 'erect':
                        if 'erect' in jpg:
                            self.total_images[jpg] = 'label'
                        else:
                            pass
                    elif self.view == 'supine':
                        if 'supine' in jpg:
                            self.total_images[jpg] = 'label'
                        else:
                            pass
                    elif self.view == 'all':
                        self.total_images[jpg] = 'label'

            if self.labeled == False:
                    p_png_lists = glob.glob(data_dir + 'unlabeled/' + '**/*.png', recursive=True)
                    p_jpg_lists = glob.glob(data_dir + 'unlabeled/' + '**/*.jpg', recursive=True)
                    for p_png in p_png_lists:
                        if self.view == 'erect':
                            if 'erect' in p_png:
                                self.total_images[p_png] = 'pseudo'
                            else:
                                pass
                        elif self.view == 'supine':
                            if 'supine' in p_png:
                                self.total_images[p_png] = 'pseudo'
                            else:
                                pass
                        elif self.view == 'all':
                            self.total_images[p_png] = 'pseudo'
                    for p_jpg in p_jpg_lists:
                        if self.view == 'erect':
                            if 'erect' in p_jpg:
                                self.total_images[p_jpg] = 'pseudo'
                            else:
                                pass
                        elif self.view == 'supine':
                            if 'supine' in p_jpg:
                                self.total_images[p_jpg] = 'pseudo'
                            else:
                                pass
                        elif self.view == 'all':
                            self.total_images[p_jpg] = 'pseudo'

        elif self.mode == 'test':   # 해당 폴더 하위를 통째로 불러옴.
            png_lists = glob.glob(data_dir + '**/*.png', recursive=True)
            jpg_lists = glob.glob(data_dir + '**/*.jpg', recursive=True)
            for png in png_lists:
                if self.view == 'erect':
                    if 'erect' in png:
                        self.total_images[png] = 'label'
                    else:
                        pass
                elif self.view == 'supine':
                    if 'supine' in png:
                        self.total_images[png] = 'label'
                    else:
                        pass
                elif self.view == 'all':
                    self.total_images[png] = 'label'

            for jpg in jpg_lists:
                if self.view == 'erect':
                    if 'erect' in jpg:
                        self.total_images[jpg] = 'label'
                    else:
                        pass
                elif self.view == 'supine':
                    if 'supine' in jpg:
                        self.total_images[jpg] = 'label'
                    else:
                        pass
                elif self.view == 'all':
                    self.total_images[jpg] = 'label'

        # # Split certain percentage of data
        # self.indices = list(range(len(self.total_images)))
        # np.random.shuffle(self.indices)
        #
        # self.selected_indices = self.indices[:int(np.floor(size * len(self.indices)))]

        self.total_images_list = sorted(self.total_images.keys())

        # self.selected_images = []
        # for index in self.selected_indices:
        #     self.selected_images.append(self.total_images_list[index])

        print('A total of %d image data were generated.' % len(self.total_images_list))
        print('View mode: {}'.format(self.view))
        self.n_data = len(self.total_images_list)

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.total_images_list[index]
        image = cv2.imread(img_path, 1)

        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        image = Image.fromarray(image)

        # Apply DINO Augmentation
        if not self.transforms == None:
            images = self.transforms(image)

        elif self.transforms == None:
            images = pth_transforms.Compose(
                [utils.GaussianBlurInference(),
                 pth_transforms.ToTensor()])(image)

        idx = img_path

        # Make label
        if 'Non-pneumoperitoneum' in idx:
            label = 0
        elif 'Pneumoperitoneum' in idx:
            label = 1
        else:
            label = 777

        if self.mode == 'train':
            return images, label, img_path
        else:
            return images, label, img_path


# class Paired_CXR_Dataset(data.Dataset):
#     'Characterizes a dataset for PyTorch'
#
#     def __init__(self, view, cfg, data_dir, size=1.0, transforms=None, mode='train', labeled=False):
#         'Initialization'
#         self.dim = (256, 256)
#         self.n_classes = 1
#         self.cfg = cfg
#         self.transforms = transforms
#         self.mode = mode
#         self.labeled = labeled
#         self.view = view
#
#         self.total_images = {}
#
#         # PNG and JPG image lists
#         if self.mode == 'train':
#             if self.labeled == True:
#                 png_lists = glob.glob(data_dir + '**/*.png', recursive=True)
#                 jpg_lists = glob.glob(data_dir + '**/*.jpg', recursive=True)
#                 for png in png_lists:
#                     if 'erect' in png:
#                         self.total_images[png] = 'label'
#                 for jpg in jpg_lists:
#                     if 'erect' in jpg:
#                         self.total_images[jpg] = 'label'
#
#             if self.labeled == False:
#                 for p_fold in self.pseudo_folds:
#                     p_png_lists = glob.glob(data_dir + '{}/'.format(p_fold) + '**/*.png', recursive=True)
#                     p_jpg_lists = glob.glob(data_dir + '{}/'.format(p_fold) + '**/*.jpg', recursive=True)
#                     for p_png in p_png_lists:
#                         self.total_images[p_png] = 'pseudo'
#                     for p_jpg in p_jpg_lists:
#                         self.total_images[p_jpg] = 'pseudo'
#
#         elif self.mode == 'test':   # 해당 폴더 하위를 통째로 불러옴.
#             png_lists = glob.glob(data_dir + '**/*.png', recursive=True)
#             jpg_lists = glob.glob(data_dir + '**/*.jpg', recursive=True)
#             for png in png_lists:
#                 if self.view == 'erect':
#                     if 'erect' in png:
#                         self.total_images[png] = 'label'
#                     else:
#                         pass
#                 elif self.view == 'supine':
#                     if 'supine' in png:
#                         self.total_images[png] = 'label'
#                     else:
#                         pass
#             for jpg in jpg_lists:
#                 if self.view == 'erect':
#                     if 'erect' in jpg:
#                         self.total_images[jpg] = 'label'
#                     else:
#                         pass
#                 elif self.view == 'supine':
#                     if 'supine' in jpg:
#                         self.total_images[jpg] = 'label'
#                     else:
#                         pass
#
#         # Split certain percentage of data
#         self.indices = list(range(len(self.total_images)))
#         np.random.seed(args.seed)
#         np.random.shuffle(self.indices)
#
#         self.selected_indices = self.indices[:int(np.floor(size * len(self.indices)))]
#
#         self.total_images_list = sorted(self.total_images.keys())
#
#         self.selected_images = []
#         for index in self.selected_indices:
#             self.selected_images.append(self.total_images_list[index])
#
#         print('A total of %d image data pairs were generated.' % len(self.selected_images))
#         self.n_data = len(self.total_images_list)
#
#         self.transforms = pth_transforms.Compose([
#             pth_transforms.RandomResizedCrop(256, scale=(0.75, 1.), interpolation=Image.BICUBIC),
#             pth_transforms.RandomHorizontalFlip(p=0.5),
#             pth_transforms.RandomRotation(degrees=(-15, 15)),
#             pth_transforms.RandomAutocontrast(p=0.3),
#             pth_transforms.RandomEqualize(p=0.3),
#             utils.GaussianBlur(0.3),
#             pth_transforms.ToTensor()
#         ])
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return self.n_data
#
#     def __getitem__(self, index):
#         'Generates one sample of data'
#
#         erect_path = self.total_images_list[index]
#         supine_path = erect_path.replace('erect', 'supine').replace('_1.png', '_2.png')
#
#         erect_image = cv2.imread(erect_path, 1)
#         supine_image = cv2.imread(supine_path, 1)
#
#         erect_image = cv2.resize(erect_image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
#         erect_image = Image.fromarray(erect_image)
#
#         supine_image = cv2.resize(supine_image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
#         supine_image = Image.fromarray(supine_image)
#
#         erect_image = pth_transforms.Compose(
#             [utils.GaussianBlurInference(),
#              pth_transforms.ToTensor()])(erect_image)
#
#         supine_image = self.transforms(supine_image)
#
#         idx = erect_path
#
#         # Make label
#         if 'Non-pneumoperitoneum' in idx:
#             label = 0
#         elif 'Pneumoperitoneum' in idx:
#             label = 1
#         else:
#             label = 777
#
#         if self.mode == 'train':
#             return (erect_image, supine_image), label
#         else:
#             return (erect_image, supine_image), label, idx