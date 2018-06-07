#import argparse
#import os
#import numpy as np
#import math
#import matplotlib.pyplot as plt
#import torchvision
#import torchvision.transforms as transforms
#from torchvision.utils import save_image

#from torch.utils.data import DataLoader, Dataset
#from torchvision import datasets
#import torchvision.datasets as dset
#from torch.autograd import Variable

import torch.nn as nn
#import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self,opt):
        self.Dropout_rate = 0.5
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        
        
        self.f2 = nn.Sequential(
            nn.Linear(122,512*4*4),
            
            )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.5),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.5),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.5),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.5),
            nn.ConvTranspose2d(64,opt.channels,4,2,1),
            nn.Tanh()
        )

    def forward(self, z, label):
        
        #out_2 = self.l1_text(label)
        combine = torch.cat([z, label],1)
        out = self.f2(combine)
        out = out.view(out.shape[0],512,4,4)
        
        img = self.conv_blocks(out)
        
        
        return img

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
    
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.5)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(534, 512, 1, 1, 0),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(512)

        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(8192, 1),
                                        nn.Sigmoid()
                                        )

    def forward(self, img, label):
        #out = self.l1(img)

        label = label.view(label.shape[0],22,1,1).repeat(1,1,4,4) 
        
        out = self.model(img)
        out = torch.cat([out,label],1)
        out = self.model2(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class GAN():
    def __init__(self,opt):
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)
        self.adversarial_loss = torch.nn.BCELoss()

        cuda = True if torch.cuda.is_available() else False

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),lr=opt.lra)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),lr = opt.lra)

