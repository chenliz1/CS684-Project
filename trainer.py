from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, network, data_path, optimizer, criterion, epoch, validation, model_filename, save_history=True):
        self.net = network
