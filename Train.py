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
    
    def train(self):
        losses = []; val_losses = []
        min_loss = np.inf; min_val_loss = np.inf 

        sum_val_loss = 0.0 
        
        self.model.eval()
        
        for data in self.val_loader:
            #Computing disparities and finding validation loss
            left = data['left_image']
            right = data['right_image']
            left = left.cuda()
            right = right.cuda()
            disps = self.model(left)
            #loss = self.loss_function(disps, [left, right])
            
            
           
            val_losses.append(loss.item())
            sum_val_loss += loss.item()
            i += 1
            
        tot_val_loss = (sum_val_loss/ (self.val_n_img * self.args.batch_size))
        print('Initial Validiation Loss:', tot_val_loss)
        
        
        
        f = open("Val_train_loss.txt", "x"); i = 0
        for e in range(self.args.epochs):
            time0 = time.time()
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
            total_train_loss = total_train_loss/(self.n_img * self.args.batch_size)
            tot_val_loss = (sum_val_loss/ (self.val_n_img * self.args.batch_size))
            f.write("Val Loss: ", tot_val_loss, "  ", "Train Loss: ", total_train_loss)
            
            if total_train_loss < min_val_loss:
                self.saveParams('params.pkl')
                min_val_loss = tot_val_loss
                print('Model_saved')

        f.close()
        print ('Min loss:', min_loss)
        self.saveParams('params.pkl')





