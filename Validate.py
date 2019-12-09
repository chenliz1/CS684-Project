import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF

class Validate(object):
    def __init__(self, val_loader, net):
        self.val_loader = val_loader
        self.net = net
        
    def __str__(self):
        return 'Validate'
    
    def validate(self):
        val_loader = self.val_loader
        net = self.net
        iou_arr = []
        net.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, masks = data

                if USE_GPU:
                    # Write me
                    inputs = inputs.cuda()
                    masks = masks.cuda()
                    net = net.cuda()
                else: 
                    # Write me
                    pass

                # Write me
                output = net(inputs)
                val_loss += MyCrossEntropyLoss(ignore_index = 255)(output, masks)
                preds = torch.argmax(output, dim = 1).cpu().numpy()

                gts = torch.from_numpy(np.array(masks.cpu(), dtype = np.int32)).long().numpy()
                gts[gts == 255] = -1
                # Hint: make sure the range of values of the ground truth is what you expect

                conf = eval_semantic_segmentation(preds, gts)

                iou_arr.append(conf['miou'])

        return val_loss, (sum(iou_arr) / len(iou_arr))