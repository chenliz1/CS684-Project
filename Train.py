import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF

class Train(object):
    def __init__(self, train_loader, net, loss_graph):
        self.train_loader = train_loader
        self.net = net
        self.loss_graph = loss_graph
        
    def __str__(self):
        return 'Train'
    
    def get_optimizer(self):
        net = self.net
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.5, nesterov=False)
        return optimizer
    
    def train(self, train_loader, net, loss_graph):
        train_loader = self.train_loader
        net = self.net
        loss_graph = self.loss_graph
        optimizer = self.get_optimizer()
        
        for i, data in enumerate(train_loader):

            inputs, masks = data

            if USE_GPU:
                inputs = inputs.cuda()
                net = net.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            main_loss = net(inputs, gts = masks)
            loss_graph.append(main_loss.item())
            main_loss.backward()
            optimizer.step()

        return main_loss





