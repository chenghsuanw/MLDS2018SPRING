import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.datasets as dset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--save_interval', type=int, default=2000, help='interval between image sampling')
parser.add_argument('--soft_label', type=bool, default=False, help='to use soft labels instead of binary ones')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def save_imgs(gen_imgs, seed):
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
    fig.savefig("./samples/gan.png")
    plt.close()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(256*ds_size**2, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class GAN():
    def __init__(self, opt):
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

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),lr=opt.lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),lr = opt.lr)


def train():
    dcgan = GAN(opt)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(loader):
            
            # Adversarial ground truths
            if not opt.soft_label:
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            else:
                valid = Variable(Tensor(np.random.uniform(0.7, 1.2, (imgs.shape[0], 1))), requires_grad=False)
                fake = Variable(Tensor(np.random.uniform(0.0, 0.3, (imgs.shape[0], 1))), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            dcgan.optimizer_G.zero_grad()

            for _ in range(1):

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                # z = Variable(Tensor(np.random.uniform(-1, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = dcgan.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = dcgan.adversarial_loss(dcgan.discriminator(gen_imgs), valid)

                g_loss.backward()
                dcgan.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            dcgan.optimizer_D.zero_grad()

            for _ in range(1):
                # Measure discriminator's ability to classify real from generated samples
                real_loss = dcgan.adversarial_loss(dcgan.discriminator(real_imgs), valid)
                fake_loss = dcgan.adversarial_loss(dcgan.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                dcgan.optimizer_D.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(loader),
                                                                d_loss.data, g_loss.data))

            # batches_done = epoch * len(loader) + i
            # if batches_done % opt.sample_interval == 0:
            #     save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

            batches_done = epoch * len(loader) + i
            if batches_done % opt.sample_interval == 0:
                np.random.seed(1010)
                z = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))
                gen_imgs = dcgan.generator(z)
                save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=5)

            if batches_done % opt.save_interval == 0:
                torch.save(dcgan.generator.state_dict(), os.path.join("./model/final_gen_"+str(batches_done)+".pkl"))
                torch.save(dcgan.discriminator.state_dict(), os.path.join("./model/final_dis_"+str(batches_done)+".pkl"))


def predict(seed):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dcgan = GAN(opt)

    dcgan.generator.load_state_dict(torch.load(gen_path))
    dcgan.discriminator.load_state_dict(torch.load(dis_path))

    np.random.seed(seed)
    z = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))
    gen_imgs = dcgan.generator(z)
  
    result = gen_imgs.data.cpu().numpy()
    ran = result.max() - result.min()
    result = (result - result.min()) / ran
    
    save_imgs(result, seed)


# dataset = dset.ImageFolder(root='./data/',
#                            transform=transforms.Compose([
#                                transforms.Resize(64),
#                                transforms.CenterCrop(64),
#                                transforms.ToTensor(),
#                            ]))
# loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
# train()


gen_path = os.path.join("./model_31/", 'final_gen_82000.pkl')
dis_path = os.path.join("./model_31/",'final_dis_82000.pkl')

predict(6972)
