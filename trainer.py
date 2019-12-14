from __future__ import absolute_import, division, print_function

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from loss import MonodepthLoss
import pickle

class Trainer:
    def __init__(self, network, train_loader, optimizer, batch_size, params_file=None, use_gpu=False):
        self.net = network
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.validator = None
        self.history = {"Train": [], "Val": []}
        self.params_file = params_file
        self.train_loader = train_loader
        self.batch_size = batch_size
        if use_gpu :
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.loss_function = MonodepthLoss(
            n=4,
            SSIM_w=0.85,
            disp_gradient_w=0.1, lr_w=1).to(self.device)


    def setValidator(self, validator):
        self.validator = validator

    def setOptimizer(self, opt):
        self.optimizer = opt

    def saveParams(self, path):
        torch.save(self.net.state_dict(), path)

    def loadModel(self, path):
        self.net.load_state_dict(torch.load(path))

    def train(self):
        total_loss = 0.0

        self.net.train()
        counter = 0
        for i, data in enumerate(self.train_loader):
            left, right = data
            if self.use_gpu:
                left = left.cuda()
                self.net = self.net.cuda()
                right = right.cuda()


            self.optimizer.zero_grad()
            disps = self.net(left)
            loss = self.loss_function(disps, [left, right])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            counter += 1

        main_loss = total_loss / counter

        return main_loss

    def run_train(self, epoch):
        if self.params_file:
            self.loadModel(self.params_file)
            save_name = "epoch{}_".format(epoch) + self.params_file
        else:
            save_name = "epoch{}_params.pkl".format(epoch)
        prev_score = np.inf
        if self.validator:
            prev_score = self.validator.validate(self.net)

        for e in range(epoch):

            loss = self.train()
            print("Epoch: {} Loss: {}".format(e, loss))
            self.history["Train"].append(loss)

            if self.validator:
                val_score = self.validator.validate(self.net)
                self.history["Val"].append(val_score)
                if val_score < prev_score:
                    print("update model file with prev_score {} and current score {}".format(prev_score, val_score))
                    self.saveParams(save_name)
                    prev_score = val_score

            with open('epoch{}_train_history.pickle'.format(epoch), 'wb') as handle:
                pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def copyNetwork(self):
        return copy.deepcopy(self.net)




