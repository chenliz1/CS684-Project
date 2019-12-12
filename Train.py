import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import numpy as np
import time
from loss import MonodepthLoss


class Train(object):
    def __init__(self, train_loader, optimizer, net, use_gpu=False, params_file=None):
        self.train_loader = train_loader
        self.model = net
        self.use_gpu = use_gpu
        self.params_file = params_file
        if use_gpu :
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.loss = MonodepthLoss(
            n=4,
            SSIM_w=0.85,
            disp_gradient_w=0.1, lr_w=1).to(self.device)
        self.optimizer = optimizer
        self.validator = None

    def setValidator(self, validation):
        self.validator = validation

    def loadModel(self, path):
        self.net.load_state_dict(torch.load(path))

    def __str__(self):
        return 'Train'
    
    def set_optimizer(self, opt):
        self.optimizer = opt


    
    def train(self):
        losses = []; val_losses = []
        min_loss = np.inf; min_val_loss = np.inf
        


        if self.params_file:
            self.loadModel(self.params_file)


        #init validation
        val_loss = self.validator.validate()
        # if self.validation:
        #     self.model.eval()
        #
        #     for i, data in self.val_loader:
        #         #Computing disparities and finding validation loss
        #         left = data['left_image']
        #         right = data['right_image']
        #         if self.use_gpu:
        #             left = left.cuda()
        #             right = right.cuda()
        #             self.model = self.model.cuda()
        #         disps = self.model(left)
        #         loss = self.loss_function(disps, [left, right])
        #
        #
        #
        #         val_losses.append(loss.item())
        #         sum_val_loss += loss.item()
        #
        #
        #     tot_val_loss = (sum_val_loss/ (self.val_n_img * self.args.batch_size))
        #     print('Initial Validiation Loss:', tot_val_loss)
        

        #training Begin

        f = open("Val_train_loss.txt", "x"); i = 0

        for e in range(self.args.epochs):

            total_train_loss = 0.0

            self.model.train()
            
            for data in self.loader:
                left = data['left_image']
                right = data['right_image']
                left = left.cuda()
                right = right.cuda()
            
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                total_train_loss += loss.item()

        total_train_loss = total_train_loss / (self.n_img * self.args.batch_size)

        if self.validation:



            #Epoch validation Begin
            sum_val_loss = 0.0


            self.model.eval()
            for data in self.val_loader:
                left = data['left_image']
                right = data['right_image']
                left = left.cuda()
                right = right.cuda()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                val_losses.append(loss.item())
                sum_val_loss += loss.item()

            # Estimate loss per image

            tot_val_loss = (sum_val_loss/ (self.val_n_img * self.args.batch_size))
            f.write("Val Loss: ", tot_val_loss, "  ", "Train Loss: ", total_train_loss)

            if total_train_loss < min_loss:
                min_loss = total_train_loss

            if tot_val_loss < min_val_loss:
                self.saveParams('params.pkl')
                min_val_loss = tot_val_loss
                print('Model_saved')

        f.close()

        print ('Min loss:', min_loss)

        self.saveParams('params.pkl')





