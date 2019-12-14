from __future__ import absolute_import, division, print_function

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from loss import MonodepthLoss

class Validator:
    def __init__(self, val_loader, batch_size, params_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.params_file = params_file
        self.val_loader = val_loader
        if use_gpu :
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.loss = MonodepthLoss(
            n=4,
            SSIM_w=0.85,
            disp_gradient_w=0.1, lr_w=1).to(self.device)
        self.val_losses = []
        self.batch_size = batch_size


    def validate(self, network):

        network.eval()

        total_loss = 0
        counter = 0
        for i, data in enumerate(self.val_loader):
            left, right = data

            if self.use_gpu:
                left = left.cuda()
                network = network.cuda()
                right = right.cuda()

            model_outputs = network(left)

            loss = self.loss(model_outputs, [left, right])
            self.val_losses.append(loss.item())
            total_loss += loss.item()
            counter += 1

        total_loss /= counter

        return total_loss







