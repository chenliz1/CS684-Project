from __future__ import absolute_import, division, print_function

import os
import glob
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import scipy
import skimage
from pypardiso import spsolve

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tF

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
class JointRandomFlip(object):
    def __call__(self, L, R):
        if np.random.random_sample()>0.5:
            return (tF.hflip(R),tF.hflip(L))
        return (L,R)
    
class JointRandomColorAug(object):

    def __init__(self,gamma=(0.8,1.2),brightness=(0.5,2.0),color_shift=(0.8,1.2)):
        self.gamma = gamma
        self.brightness = brightness
        self.color_shift = color_shift

    def __call__(self, L, R):
        if  np.random.random_sample()>0.5:
            
            random_gamma = np.random.uniform(*self.gamma)
            L_aug = L ** random_gamma
            R_aug = R ** random_gamma

            random_brightness = np.random.uniform(*self.brightness)
            L_aug = L_aug * random_brightness
            R_aug = R_aug * random_brightness

            random_colors = np.random.uniform(self.color_shift[0],self.color_shift[1], 3)
            for i in range(3):
                L_aug[i, :, :] *= random_colors[i]
                R_aug[i, :, :] *= random_colors[i]

            # saturate
            L_aug = torch.clamp(L_aug, 0, 1)
            R_aug = torch.clamp(R_aug, 0, 1)

            return L_aug, R_aug

        else:
            return L, R

class JointToTensor(object):
    def __call__(self, L, R):
        return tF.to_tensor(L),tF.to_tensor(R)
    
class JointToImage(object):
    def __call__(self, L, R):
        return transforms.ToPILImage()(L),transforms.ToPILImage()(R)
    
    
class JointCompose(object):
    def __init__(self, transforms):
        """
        params: 
           transforms (list) : list of transforms
        """
        self.transforms = transforms

    # We override the __call__ function such that this class can be
    # called as a function i.e. JointCompose(transforms)(img, target)
    # Such classes are known as "functors"
    def __call__(self, img, target):
        """
        params:
            img (PIL.Image)    : input image
            target (PIL.Image) : ground truth label 
        """
        assert img.size == target.size
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


train_joint_transform = JointCompose([JointRandomFlip(),JointToTensor(),JointRandomColorAug()])


class TwoViewDataset(data.Dataset):
    
    def __init__(self, 
                 data_path,
                 resize_shape=(512,256), 
                 is_train=False,
                 transforms=None,
                 sanity_check=None):
        super(TwoViewDataset, self).__init__()
        self.data_path = data_path

        self.interp = Image.ANTIALIAS
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.transforms=transforms
        self.loader = pil_loader
        
        if is_train:
            self.imgR_folder = os.path.join(data_path, "train", "image_right")
            self.imgL_folder = os.path.join(data_path, "train", "image_left")
        else:
            self.imgR_folder = os.path.join(data_path, "val", "image_right")
            self.imgL_folder = os.path.join(data_path, "val", "image_left")
        
        
        self.imgR=[os.path.join(self.imgR_folder, x) for x in os.listdir(self.imgR_folder)]
        self.imgL=[os.path.join(self.imgL_folder, x) for x in os.listdir(self.imgL_folder)]

    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return self.to_tensor(color)


    def __len__(self):
        return len(list(glob.glob1(self.imgL_folder, "*.jpg")))

    def __getitem__(self, index):
        #print(np.array(Image.open(self.imgR[index]).convert('RGB')).shape)
        colorR=Image.open(self.imgR[index]).convert('RGB').resize(self.resize_shape)
        colorL=Image.open(self.imgL[index]).convert('RGB').resize(self.resize_shape)
        #print(np.array(colorR).shape)
        
        if self.transforms is not None:
            colorR, colorL = self.transforms(colorR, colorL)
        return colorL, colorR


class GroundTruth(data.Dataset):
    def __init__(self, 
                 data_path,
                 resize_shape=(512,256), 
                 is_train=False,
                 transforms=JointToTensor(),
                 sanity_check=None):
        super(GroundTruth, self).__init__()
        self.data_path = data_path

        self.interp = Image.ANTIALIAS
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.transforms=transforms
        self.loader = pil_loader
        
        self.imgR_folder = os.path.join(data_path, "val", "image_right")
        self.imgL_folder = os.path.join(data_path, "val", "image_left")
        self.gtR_folder = os.path.join(data_path, "val", "gt_right")
        self.gtL_folder = os.path.join(data_path, "val", "gt_left")
        
        self.imgR=[os.path.join(self.imgR_folder, x) for x in os.listdir(self.imgR_folder)]
        self.imgL=[os.path.join(self.imgL_folder, x) for x in os.listdir(self.imgL_folder)]
        self.gtR=[os.path.join(self.gtR_folder, x) for x in os.listdir(self.gtR_folder)]
        self.gtL=[os.path.join(self.gtL_folder, x) for x in os.listdir(self.gtL_folder)]

    def depth_read(self, filename):

        depth_png = np.array(Image.open(filename).resize(self.resize_shape), dtype=int)
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        return depth


    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return self.to_tensor(color)

    def depth_colorization(self, imgRgb=None, imgDepthInput=None, alpha=1):
        imgIsNoise = imgDepthInput == 0
        maxImgAbsDepth = np.max(imgDepthInput)
        imgDepth = imgDepthInput / maxImgAbsDepth
        imgDepth[imgDepth > 1] = 1
        (H, W) = imgDepth.shape
        numPix = H * W
        indsM = np.arange(numPix).reshape((W, H)).transpose()
        knownValMask = (imgIsNoise == False).astype(int)
        grayImg = skimage.color.rgb2gray(imgRgb)
        winRad = 1
        len_ = 0
        absImgNdx = 0
        len_window = (2 * winRad + 1) ** 2
        len_zeros = numPix * len_window

        cols = np.zeros(len_zeros) - 1
        rows = np.zeros(len_zeros) - 1
        vals = np.zeros(len_zeros) - 1
        gvals = np.zeros(len_window) - 1

        for j in range(W):
            for i in range(H):
                nWin = 0
                for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                    for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                        if ii == i and jj == j:
                            continue

                        rows[len_] = absImgNdx
                        cols[len_] = indsM[ii, jj]
                        gvals[nWin] = grayImg[ii, jj]

                        len_ = len_ + 1
                        nWin = nWin + 1

                curVal = grayImg[i, j]
                gvals[nWin] = curVal
                c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

                csig = c_var * 0.6
                mgv = np.min((gvals[:nWin] - curVal) ** 2)
                if csig < -mgv / np.log(0.01):
                    csig = -mgv / np.log(0.01)

                if csig < 2e-06:
                    csig = 2e-06

                gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
                gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
                vals[len_ - nWin:len_] = -gvals[:nWin]

                # Now the self-reference (along the diagonal).
                rows[len_] = absImgNdx
                cols[len_] = absImgNdx
                vals[len_] = 1  # sum(gvals(1:nWin))

                len_ = len_ + 1
                absImgNdx = absImgNdx + 1

        vals = vals[:len_]
        cols = cols[:len_]
        rows = rows[:len_]
        A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

        rows = np.arange(0, numPix)
        cols = np.arange(0, numPix)
        vals = (knownValMask * alpha).transpose().reshape(numPix)
        G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

        A = A + G
        b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

        #print ('Solving system..')

        new_vals = spsolve(A, b)
        new_vals = np.reshape(new_vals, (H, W), 'F')

        #print ('Done.')

        denoisedDepthImg = new_vals * maxImgAbsDepth
        
        output = denoisedDepthImg.reshape((H, W)).astype('float32')

        output = np.multiply(output, (1-knownValMask)) + imgDepthInput
        
        return output

    def __len__(self):
        return len(list(glob.glob1(self.imgL_folder, "*.jpg")))

    def __getitem__(self, index):
        #print(np.array(Image.open(self.imgR[index]).convert('RGB')).shape)
        colorR=Image.open(self.imgR[index]).convert('RGB').resize(self.resize_shape)
        colorL=Image.open(self.imgL[index]).convert('RGB').resize(self.resize_shape)
        #print(np.array(colorR).shape)
        gtR=self.depth_read(self.gtR[index])
        gtL=self.depth_read(self.gtL[index])


        colorR_N = np.array(colorR)/255.0
        colorL_N = np.array(colorL)/255.0


        gtR = self.depth_colorization(imgRgb=colorR_N, imgDepthInput=gtR, alpha=1)
        gtL = self.depth_colorization(imgRgb=colorL_N, imgDepthInput=gtL, alpha=1)
        
        # if self.transforms is not None:
        #     colorR, colorL = self.transforms(colorR, colorL)
        #     gtR, gtL = self.transforms(gtR, gtL)


        return gtL, gtR
