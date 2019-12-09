from __future__ import absolute_import, division, print_function

import os
import glob
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class depthDataset(data.Dataset):
    def __init__(self, data_path, resize_shape=(1242, 375), is_train=False):
        super(depthDataset, self).__init__()
        self.data_path = data_path

        self.interp = Image.ANTIALIAS
        self.resize_shape = resize_shape
        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        if is_train:
            self.gts_folder = os.path.join(data_path, "train", "groundtruth")
            self.img_folder = os.path.join(data_path, "train", "image")
        else:
            self.gts_folder = os.path.join(data_path, "val", "groundtruth")
            self.img_folder = os.path.join(data_path, "val", "image")



    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return self.to_tensor(color)



    def get_depth(self, path, do_flip):
        depth_png = Image.open(path)

        depth_png = depth_png.resize(self.resize_shape, Image.NEAREST)
        if do_flip:
            depth_png = depth_png.transpose(Image.FLIP_LEFT_RIGHT)
        depth =  np.array(depth_png, dtype = int)

        assert (np.max(depth_png) > 255)
        depth = np.array(depth_png).astype(np.float) / 256.
        depth[depth_png == 0] = -1.


        return torch.from_numpy(depth.astype(np.float32))

    def __len__(self):
        return len(list(glob.glob1(self.gts_folder, "*.png")))

    def __getitem__(self, index):

        gts_fn = os.path.join(self.gts_folder, str(index).zfill(5) + ".png")
        img_fn = os.path.join(self.img_folder, str(index).zfill(5) + ".jpg")

        do_flip = self.is_train and random.random() > 0.5

        color = self.get_color(img_fn, do_flip)
        depth = self.get_depth(gts_fn, do_flip)

        return color, depth





