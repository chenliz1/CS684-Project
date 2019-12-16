from __future__ import absolute_import, division, print_function

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tF

def postprocess(image, network): #input is PIL image

    spF = tF.to_tensor(tF.hflip(image)).unsqueeze(0)
    sp = tF.to_tensor(image).unsqueeze(0)
    disp = network(sp)[0][0][0].detach().numpy()
    f_disp = network(spF)[0][0][0].detach().numpy()
    width = disp.shape[-1]
    height = disp.shape[-2]
    dl = disp
    d_l = np.fliplr(f_disp)
    wl = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for j in range(width):
            if (j / width) <= 0.1:
                wl[i, j] = 1.0
            elif (j / width) > 0.2:
                wl[i, j] = 0.5
            else:
                wl[i, j] = 5 * (0.2 - (j / width)) + 0.5

    w_l = np.fliplr(wl)

    return dl * wl + d_l * w_l


#sample = TwoViewDataset("data/dataset/", is_train=False)[index of sample][0: left or 1: right]
#sample = TwoViewDataset("data/dataset/", is_train=False)[55][0]
#disp_np = postprocess(sample, network)

