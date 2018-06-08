import argparse
import os
import numpy as np
#import math
import matplotlib.pyplot as plt
#import torchvision
#import torchvision.transforms as transforms
from torchvision.utils import save_image

#from torch.utils.data import DataLoader, Dataset
#from torchvision import datasets
#import torchvision.datasets as dset
from torch.autograd import Variable

#import torch.nn as nn
#import torch.nn.functional as F
import torch
from dcgan_32 import GAN
#import pandas as pd
from Custom_data_32 import FaceWithFeature
import pickle
import sys



argv=sys.argv[2:]
tags = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lra', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lrr', type=float, default=0.00001, help='rmsprop: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--save_interval', type=int, default=2000, help='interval between image sampling')
opt = parser.parse_args(argv)
print(opt)

cuda = True if torch.cuda.is_available() else False

hair_features = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
 'green hair', 'red hair', 'purple hair', 'pink hair',
  'blue hair', 'black hair', 'brown hair', 'blonde hair']
eyes_features = ['black eyes', 'orange eyes',
 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

hfc = [0]*12
efc = [0]*10

def save_imgs(gen_imgs,seed):

    r, c = 5, 5
    gen_imgs = np.swapaxes(gen_imgs,1,3)
    gen_imgs = np.swapaxes(gen_imgs,2,1)
    gen_imgs[gen_imgs<0] = 0
    
    

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./samples/cgan.png")
    plt.close()

def get_sample_feature(tags_v):
    
    f = open(tags_v)
    result = []
    for lines in f.readlines():
        re = [0]*22
        l = lines.strip().split(",")
        k = l[1].split(" ")
    
        hair = " ".join(k[:2])
        
        eyes = " ".join(k[-2:])
        re[hair_features.index(hair)] = 1
        re[eyes_features.index(eyes)+12] = 1
        result.append(re)
    #for i in range(opt.batch_size-25):
    #    result.append(re)
    return result
        


def train():
    
    
    
    dcgan = GAN(opt)



    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    test_sample = get_sample_feature()
    test_sample = Variable(Tensor(test_sample))

    # ----------
    #  Training
    # ----------

    

    for epoch in range(opt.n_epochs):
        for i, (imgs,w_imgs, feature,f_feature) in enumerate(loader):
            # Adversarial ground truths
            
            valid = np.random.rand(imgs.shape[0])/5 + 0.9
            fake = np.random.rand(imgs.shape[0])/5
            valid = Variable(Tensor(valid))
            fake = Variable(Tensor(fake))
            
            
            
            valid_1 = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            #fake_0 = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            
            
            f_feature = Variable(Tensor(f_feature))
            
            
            
            
            feature = Variable(feature)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            wrong_imgs = Variable(w_imgs.type(Tensor))
            #noise = Variable(real_imgs.data.new(real_imgs.size()).normal_(0, stddev))
            #stddev *= 0.995
            #real_imgs = real_imgs + noise
            # -----------------
            #  Train Generator
            # -----------------

            dcgan.optimizer_G.zero_grad()
            

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            
            
            

            # Generate a batch of images
            gen_imgs = dcgan.generator(z,feature)
            
            



            # Loss measures generator's ability to fool the discriminator
            #non-flipped generator label
            g_loss = dcgan.adversarial_loss(dcgan.discriminator(gen_imgs, feature), valid_1)


            #flipped generator label
            #g_loss = dcgan.adversarial_loss(dcgan.discriminator(gen_imgs, feature), fake)
            
            
            
            
            #g_loss = -torch.mean(dcgan.discriminator(gen_imgs, feature))


        
            
            g_loss.backward()
            dcgan.optimizer_G.step()
            

            
            

            # ---------------------
            #  Train Discriminator
            # ---------------------

            times = 1 if epoch > 50 else 3

            for _ in range(times):
                dcgan.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                
                real_loss = dcgan.adversarial_loss(dcgan.discriminator(real_imgs, feature), valid)
                fake_loss = dcgan.adversarial_loss(dcgan.discriminator(gen_imgs.detach(), feature), fake)
                fake_loss2 = dcgan.adversarial_loss(dcgan.discriminator(real_imgs, f_feature), fake)
                fake_loss3 = dcgan.adversarial_loss(dcgan.discriminator(wrong_imgs, feature), fake)
                d_loss = (real_loss + (fake_loss + fake_loss2 + fake_loss3) / 3)
                
                '''
                real_loss = torch.mean(dcgan.discriminator(real_imgs, feature))
                fake_loss = torch.mean(dcgan.discriminator(gen_imgs.detach(), feature))
                fake_loss2 = torch.mean(dcgan.discriminator(real_imgs, f_feature))
                fake_loss3 = torch.mean(dcgan.discriminator(wrong_imgs, feature))


                d_loss = -(real_loss-fake_loss-fake_loss2-fake_loss3) / 4
                '''

                LOSS = float(d_loss)
                if LOSS > 0.5:
                    d_loss.backward()
                    dcgan.optimizer_D.step()
            '''
            for p in dcgan.discriminator.parameters():
                p.data.clamp_(-0.01,0.01)
            '''
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(loader),
                                                                d_loss.data, g_loss.data))

            batches_done = epoch * len(loader) + i
            if batches_done % opt.sample_interval == 0:
                np.random.seed(7)
                z = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))
                gen_imgs = dcgan.generator(z,test_sample)
                torch.div(gen_imgs,2)
                torch.add(gen_imgs,1)
                save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=5)

            if batches_done % opt.save_interval == 0:
                torch.save(dcgan.generator.state_dict(), os.path.join("./model/final_gen_"+str(batches_done)+".pkl"))
                torch.save(dcgan.discriminator.state_dict(), os.path.join("./model/final_dis_"+str(batches_done)+".pkl"))

    np.random.seed(7)
    #z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
    z = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))
    gen_imgs = dcgan.generator(z,test_sample)
    #save_image(gen_imgs.data, 'images/final.png', nrow=5, normalize=True)
    
    torch.save(dcgan.generator.state_dict(), os.path.join("./model/", 'final_gen.pkl'))
    torch.save(dcgan.discriminator.state_dict(), os.path.join("./model/",'final_dis.pkl'))
    exit()

def predict(seed, tags):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    test_sample = get_sample_feature(tags)
    size = len(test_sample)
    test_sample = Variable(Tensor(test_sample))
    dcgan = GAN(opt)
    
    

    dcgan.generator.load_state_dict(torch.load(gen_path))
    dcgan.discriminator.load_state_dict(torch.load(dis_path))


    #np.random.seed(seed)
    #z = np.random.normal(0, 1, (size, opt.latent_dim))
    #np.savetxt("z_32.txt",z)
    #exit()
    z = np.loadtxt("z_32.txt")
    z = Variable(Tensor(z))
    gen_imgs = dcgan.generator(z,test_sample)
    
    #gen_imgs = torch.div(gen_imgs,2)
    #gen_imgs = torch.add(gen_imgs,0.5)
    #gen_imgs = torch.mul(gen_imgs,255)
    
    result = gen_imgs.data.cpu().numpy()
    
    
    save_imgs(result, seed)
    #exit()
    #save_image(gen_imgs.data[:25], 'images/final_predict.png', nrow=5)

def load_tags():
    face = []
    extra = []
    faces_tags = pd.read_csv("./AnimeDataset/tags_clean.csv", header=None)
    extra_tags = pd.read_csv("./extra_data/tags.csv", header= None)
    for i in range(faces_tags.shape[0]):
        
        feature = faces_tags[1][i].lower()
        ht = False
        et = False
        ht_c = 0
        et_c = 0
        for h in hair_features:
            if h in feature:
                ht = h
                ht_c += 1
        for e in eyes_features:
            if e in feature:
                et = e
                et_c += 1
        if ht_c == 1 and et_c == 1:
            hfc[hair_features.index(ht)]+=1
            efc[eyes_features.index(et)]+=1
            face.append([i,ht,et])
    for i in range(extra_tags.shape[0]):
        feature = extra_tags[1][i].lower().split(" ")
        extra.append([i," ".join(feature[:2])," ".join(feature[-2:])])
    
    for i in range(len(hair_features)):
        print(hair_features[i],hfc[i])

    for i in range(len(eyes_features)):
        print(eyes_features[i],efc[i])
    print(sum(hfc))
    print(sum(efc))
    return face, extra
    



#face, extra = load_tags()
    


gen_path = os.path.join("./model_32/", 'final_gen_102000.pkl')
dis_path = os.path.join("./model_32/",'final_dis_102000.pkl')
'''
face_dataset = dset.ImageFolder(root='./AnimeDataset',
                           transform=transforms.Compose([
                               transforms.Resize(64),
#                               transforms.CenterCrop(96),
                               transforms.ToTensor(),
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

extra_dataset = dset.ImageFolder(root='./extra_data',
                           transform=transforms.Compose([
                               transforms.Resize(64),
#                               transforms.CenterCrop(96),
                               transforms.ToTensor(),
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
'''
'''
dataset = FaceWithFeature('./AnimeDataset/faces',"./extra_data/images",face,extra,
                          transform=transforms.Compose([
                               transforms.Resize(64),
#                               transforms.CenterCrop(96),
                               
                               transforms.RandomHorizontalFlip(0.5),
                               #transforms.RandomRotation(20),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))




loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

#train()
'''

predict(34, tags)


