from __future__ import absolute_import, division, print_function

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, network, train_loader, optimizer, params_file=None, use_gpu=True):
        self.net = network
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.validation = None
        self.history = []
        self.params_file = params_file
        self.train_loader = train_loader


    def setValidation(self, validation):
        self.validation = validation

    def saveParams(self, path):
        torch.save(self.net.state_dict(), path)

    def loadModel(self, path):
        self.net.load_state_dict(torch.load(path))

    def train(self):

        self.net.train()
        for i, data in enumerate(self.train_loader):
            left, right = data
            if self.use_gpu:
                left = left.cuda()
                self.net = self.net.cuda()
                right = right.cuda()
            self.optimizer.zero_grad()
            main_loss = net(left, right_view=right)
            self.history.append(main_loss.item())
            main_loss.backward()
            self.optimizer.step()

        return main_loss

    def run_train(self, epoch):
        if self.params_file:
            self.loadModel(self.params_file)
        if self.validation:
            prev_score = self.validation.validate(self.net)
        for e in range(epoch):
            loss = self.train()
            print("Epoch: {} Loss: {}".format(e, loss))
            if self.validation:
                val_score = self.validation.validate(self.net)
                if val_score > prev_score:
                    print("update model file with prev_score {} and current score {}".format(prev_score, val_score))
                    self.saveParams('params.pkl')
                    prev_score = val_score


    def copyNetwork(self):
        return copy.deepcopy(self.net)




