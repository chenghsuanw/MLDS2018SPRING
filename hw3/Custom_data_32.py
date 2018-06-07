import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.datasets as dset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from dcgan_32 import GAN
import pandas as pd
import pickle
from PIL import Image
import random

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

hf = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
 'green hair', 'red hair', 'purple hair', 'pink hair',
  'blue hair', 'black hair', 'brown hair', 'blonde hair']
ef = ['black eyes', 'orange eyes',
 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']


class FaceWithFeature(Dataset):
    
    def __init__(self, face_dir ,extra_dir, face_f, extra_f,transform):
        hfd = dict()
        efd = dict()
        for i in range(len(hf)):
            hfd[hf[i]] = i
        for i in range(len(ef)):
            efd[ef[i]] = i

        tag_list = []
        tags_list = dict()

        for i in range(12):
            for j in range(10):
                tag = str(i)+str(j)
                tag_list.append(tag)
                tags_list[tag] = []


        data = []
        self.Data = []
        self.transform = transform
        for ff in face_f:
            index, hair_f, eyes_f = ff
            label = [0]*22
            label_fake = [0]*22
            h_offest = 0
            e_offset = 0
            tag = str(hfd[hair_f])+str(efd[eyes_f])
            tags_list[tag].append(face_dir+"/"+str(index)+".jpg")
            while h_offest == 0 and e_offset == 0:
                h_offest = random.randint(0,11) 
                e_offset = random.randint(0,9) 

            label[hfd[hair_f]] = 1
            label[efd[eyes_f]+12] = 1
            label_fake[ (hfd[hair_f]+h_offest)%12 ] = 1
            label_fake[ (efd[eyes_f]+e_offset)%10 + 12] = 1
            #im = Image.open(face_dir+"/"+str(index)+".jpg")
            data.append([face_dir+"/"+str(index)+".jpg",label,label_fake,tag])
            
        for ee in extra_f:
            index, hair_f, eyes_f = ee
            label = [0]*22
            label_fake = [0]*22
            h_offest = 0
            e_offset = 0
            tag = str(hfd[hair_f])+str(efd[eyes_f])
            tags_list[tag].append(extra_dir+"/"+str(index)+".jpg")
            while h_offest == 0 and e_offset == 0:
                h_offest = random.randint(0,11) 
                e_offset = random.randint(0,9) 

            label[hfd[hair_f]] = 1
            label[efd[eyes_f]+12] = 1
            label_fake[ (hfd[hair_f]+h_offest)%12 ] = 1
            label_fake[ (efd[eyes_f]+e_offset)%10 + 12] = 1
            
            data.append([extra_dir+"/"+str(index)+".jpg",label,label_fake,tag])

        for d in data:
            real_data = d[0]
            label = d[1]
            wrong_label = d[2]
            tag = d[3]
            ran = tag_list.index(tag)
            while ran == tag_list.index(tag):
                ran = random.randint(0,len(tag_list)-1)
            rann = random.randint(0,len(tags_list[tag_list[ran]])-1)
            wrong_image = tags_list[tag_list[ran]][rann]
            self.Data.append([real_data, wrong_image, label, wrong_label])


    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        im = Image.open(self.Data[index][0])
        
        w_im = Image.open(self.Data[index][1])
        label = self.Data[index][2]
        w_label = self.Data[index][3]
        
        return [self.transform(im),self.transform(w_im),Tensor(label),Tensor(w_label)]

            
