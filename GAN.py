# -*- coding: utf-8 -*-
'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-24 14:56:49
LastEditors: jinlong li
LastEditTime: 2020-09-12 09:41:06
'''
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pdb

from utils import weight_init, load_data
# from function import Data_statistics


''' Generator'''

'''1.G_CelebA: '''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.ngpu = ngpu
        self.nz = 100
        self.ngf = 64
        self.nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


''' 2. G_myRectangle: '''
class My_G(nn.Module):
    def __init__(self, nz):
        super(My_G, self).__init__()
        # self.ngpu = ngpu
        self.nz = nz
        self.ngf = 128
        self.nc = 3
        # self.main = nn.Sequential(
        #     # input is Z= 100, going into a convolution  
        #     nn.ConvTranspose2d( self.nz, self.ngf * 16, (1,5), 1, 0, bias=False), #(ngf*16) x 1 x 5
        #     nn.BatchNorm2d(self.ngf * 16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, (5,4), 4, 0, bias=False), #(ngf*8) x 5 x 20 
        #     nn.BatchNorm2d(self.ngf * 8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 8, self.ngf * 4, (3,4), 2, 1, bias=False), #(ngf*4) x 9 x 40
        #     nn.BatchNorm2d(self.ngf * 4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 4, 0, bias=False),  #(ngf) x 68 x 320 
        #     nn.BatchNorm2d(self.ngf),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf, self.nc, (2,4), 4, 0, bias=False),  #(nc) x 270 x 1280
        #     nn.Tanh()
        # )
        # self.main = nn.Sequential(
        #     # input is Z=100, going into a convolution
        #     nn.ConvTranspose2d( self.nz, self.ngf * 8, (1,5), 1, 0, bias=False), #(ngf*16) x 1 x 5
        #     nn.BatchNorm2d(self.ngf * 8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, (4,4), 2, 1, bias=False), #(ngf*8) x 2 x 10 
        #     nn.BatchNorm2d(self.ngf * 8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 8, self.ngf * 4, (5 ,4), 2, 1, bias=False), #(ngf*4) x 5 x 20
        #     nn.BatchNorm2d(self.ngf * 4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 4, (4,4), 2, 1, bias=False),  #(ngf*2) x 10 x 40
        #     nn.BatchNorm2d(self.ngf * 4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 4, (4,4) ,2, 1, bias=False),  #(ngf) x 20 x 80 
        #     nn.BatchNorm2d(self.ngf *4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf *4, self.ngf * 2, (4,4) ,2, 1, bias=False),  #(ngf) x 40 x 160 
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf *2, self.ngf * 2, (4,4) ,2, 1, bias=False),  #(ngf) x 80 x 320 
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf * 2, self.ngf, (4,4) ,2, 1, bias=False),  #(ngf) x 160 x 640
        #     nn.BatchNorm2d(self.ngf),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d( self.ngf, self.nc, (4,4), 2, 1, bias=False),  #(nc) x 320 x 1280
        #     nn.Tanh()
        # )
        # self.conv1 = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d( self.nz, self.ngf * 16, (1,5), 1, 0, bias=False), #(ngf*16) x 1 x 5
        #     nn.BatchNorm2d(self.ngf * 16),
        #     nn.ReLU(True),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, (5,4), 4, 0, bias=False), #(ngf*8) x 5 x 20 
        #     nn.BatchNorm2d(self.ngf * 8),
        #     nn.ReLU(True),            
        # )
        # self.conv3 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 8, self.ngf * 4, (3,4), 2, 1, bias=False), #(ngf*4) x 9 x 40
        #     nn.BatchNorm2d(self.ngf * 4),
        #     nn.ReLU(True),            
        # )
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        # )
        # self.conv5 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, (3,4), 2, 1, bias=False),  #(ngf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        # )
        self.main = nn.Sequential(
            # input is Z=100, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 16, (5,5), 1, 0, bias=False), #(ngf*16) x 5 x 5
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, (4,4), 2, 1, bias=False), #(ngf*8) x 10 x 10 
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 8, self.ngf * 4, (4 ,4), 2, 1, bias=False), #(ngf*4) x 20 x 20
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 4, (4,4), 2, 1, bias=False),  #(ngf*2) x 40 x 40
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 4, (4,4) ,2, 1, bias=False),  #(ngf) x 80 x 80 
            nn.BatchNorm2d(self.ngf *4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf *4, self.ngf * 4, (4,4) ,2, 1, bias=False),  #(ngf) x 160 x 160 
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf *4, self.ngf * 2, (4,4) ,2, 1, bias=False),  #(ngf) x 320 x 320 
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, (4,4) ,2, 1, bias=False),  #(ngf) x 640 x 640
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf, self.nc, (4,4), 2, 1, bias=False),  #(nc) x 1280 x 1280
            nn.Tanh()
        )
        # self.main = nn.Sequential(
        #     # input is Z=100, going into a convolution
        #     nn.ConvTranspose2d(self.nz, self.ngf*16, 5, 1, 0, bias=False),
        #     #(ngf*16) x 5 x 5
        #     nn.BatchNorm2d(self.ngf*16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf*16, self.ngf*16, 4, 2, 1, bias=False),
        #     #(ngf*16) x 10 x 10
        #     nn.BatchNorm2d(self.ngf*16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf*16, self.ngf*16, 4, 2, 1, bias=False),
        #     #(ngf*16) x 20 x 20
        #     nn.BatchNorm2d(self.ngf*16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf*16, self.ngf*8, 6, 4, 1, bias=False),
        #     #(ngf*8) x 80 x 80
        #     nn.BatchNorm2d(self.ngf*8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 6, 4, 1, bias=False),
        #     #(ngf*4) x 320 x 320
        #     nn.BatchNorm2d(self.ngf*4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(self.ngf*4, self.nc, 6, 4, 1, bias=False),
        #     #(ngf*2) x 1280 x 320            
        #     nn.Tanh()

        # )
        

    def forward(self, input):
        return self.main(input)



'''Discriminator'''

''' 1.D_CelebA: '''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.ndf = 64
        self.nc = 3
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

''' 2.G_myRectangle:'''
class My_D(nn.Module):
    def __init__(self):
        super(My_D, self).__init__()
        # self.ngpu = ngpu
        self.ndf = 64
        self.nc = 3
        # self.main = nn.Sequential(
        #     # input is (nc) x 270 x 1280
        #     nn.Conv2d(self.nc, self.ndf, (2,4), 4, 0, bias=False), # (ndf) x 68 x 320
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf, self.ndf * 2, (4,4), 4, 0, bias=False), #(ndf*2) x 17 x 80
        #     nn.BatchNorm2d(self.ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 2, self.ndf * 4, (1,4), 2, 1, bias=False), # (ndf*4) x 9 x 40
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 8, (1,4), 2, 1, bias=False), # (ndf*8) x 5 x 20
        #     nn.BatchNorm2d(self.ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 8, self.ndf * 16, (1,4), 4, 0, bias=False), # (ndf*16) x 2 x 5
        #     nn.BatchNorm2d(self.ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 16, 1, (2,5), 1, 0, bias=False), # 1 x 1 x 1
        #     nn.Sigmoid()
        # )
        # self.main = nn.Sequential(
        #     # input is (nc) x 320 x 1280
        #     nn.Conv2d(self.nc, self.ndf, (4,4), 4, 0, bias=False), # (ndf) x 80 x 320
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf, self.ndf * 2, (4,4), 4, 0, bias=False), #(ndf*2) x 20 x 80
        #     nn.BatchNorm2d(self.ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 2, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 10 x 40
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 8, (4,4), 2, 1, bias=False), # (ndf*8) x 5 x 20
        #     nn.BatchNorm2d(self.ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 8, self.ndf * 16, (1,4), 4, 0, bias=False), # (ndf*16) x 2 x 5
        #     nn.BatchNorm2d(self.ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 16, 1, (2,5), 1, 0, bias=False), # 1 x 1 x 1
        #     nn.Sigmoid()
        # )
        self.main = nn.Sequential(
            # input is (nc) x 1280 x 1280
            nn.Conv2d(self.nc, self.ndf, (4,4), 2, 1, bias=False), # (ndf) x 640 x 640
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, (4,4), 2, 1, bias=False), #(ndf*2) x 320 x 320
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 160 x 160
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 80 x 80
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 40 x 40
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 20 x 20
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, (4,4), 2, 1, bias=False), # (ndf*4) x 10 x 10
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, self.ndf * 16, (4,4), 2, 1, bias=False), # (ndf*16) x 5 x 5
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 16, 1, (5,5), 1, 0, bias=False), # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


'''
DCGAN training
'''
class DCGAN_Trainer():
    def __init__(self,G, D):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = G
        self.D = D
        self.loss = []
        self.image_size = 64
        self.batch_size =64
        self.worker = 2
        self.dataroot = '/home/admin11/Data_test/test_1000'#'/home/admin11/Data_test/celeba'
        self.lr = 0.0002
        self.epoch = 500
        self.beta1 = 0.5
        self.nz = 100

        self.img_list = []
        self.G_loss = []
        self.D_loss = []
        

        self.img_path = '/home/admin11/1.my_zone/out_imgs/1.DCGAN/CelebA/imgs'
        self.model_path = '/home/admin11/1.my_zone/model/1.DCGAN/CelebA/'
        self.type ='demo'#'CelebA'

    def training(self, dataloader):
        self.G.to(self.device)
        self.D.to(self.device)
        self.G.apply(weight_init)
        self.D.apply(weight_init)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        # test_path = '/home/admin11/1.my_zone/test_imgs'
        # types = 'AE_CelebA'
        # types = 'demo_1000'
        # img_dis = np.loadtxt(os.path.join(test_path, str(types)+'_AE.txt'))
        # Cal_img = Data_statistics(img_dis)
        # fixed_noise = Cal_img.cal(self.batch_size)
        # fixed_noise = fixed_noise.cuda()
        
        

        fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
        

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        if not os.path.exists(self.img_path+'/'+self.type):
            os.makedirs(self.img_path+'/'+self.type)
        if not os.path.exists(self.model_path+'/'+self.type):
            os.makedirs(self.model_path+'/'+self.type)
        if not os.path.exists(self.img_path+'/DCGAN_log'):
            os.makedirs(self.img_path+'/DCGAN_log')
        writer = SummaryWriter(os.path.join(self.img_path, 'DCGAN_log'))

        iters = 0

        print("Starting Training Loop...")
        for epoch in range(self.epoch):

            for i, (data, _) in enumerate(dataloader, 0):

                iters +=1

                self.D.zero_grad()
                real_img = data.to(self.device)
                bs = real_img.size(0)
                label = torch.full((self.batch_size,), real_label, device=self.device)
                # Forward pass real batch through D
                output = self.D(real_img).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
                # noise = Cal_img.cal(self.batch_size)
                # noise = noise.cuda()
                
                # Generate fake image batch with G
                fake = self.G(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.G.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.D(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch+1, self.epoch, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_loss.append(errG.item())
                self.D_loss.append(errD.item())
                writer.add_scalar('D_Loss',errD.item(), iters)
                writer.add_scalar('G_Loss',errG.item(), iters)

                # Check how the generator is doing by saving G's output on fixed_noise
                if (i == len(dataloader)-1):
                    with torch.no_grad():
                        fake_img = self.G(fixed_noise).detach().cpu()
                        save_image(fake, os.path.join(os.path.join(self.img_path, self.type), str(self.type)+'_dcgan_%d.png'%(epoch+1)), normalize=True)
                        writer.add_image('generation_images', fake_img[0])
                    self.img_list.append(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True))

        #save the net
        state_1 = {'net':self.G.state_dict(), 'optimizer':optimizerG.state_dict(), 'epoch':epoch}
        torch.save(state_1, os.path.join(os.path.join(self.model_path, self.type),str(self.type)+'_G_dcgan_%d.pth' %(epoch+1)))
        state_2 = {'net':self.D.state_dict(), 'optimizer':optimizerD.state_dict(), 'epoch':epoch}
        torch.save(state_2, os.path.join(os.path.join(self.model_path, self.type), str(self.type)+'_D_dcgan_%d.pth'%(epoch+1)))

        plt.figure(figsize=(10,5))
        plt.title(str(self.type) + "_DCGAN Loss During Training")
        plt.plot(self.D_loss, label="D_Loss")
        plt.plot(self.G_loss, label="G_Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.model_path, self.type), str(self.type) + '_dcgan_%03d_epoch.png'%(self.epoch)))

        # Animation showing the improvements of the generator.
        fig = plt.figure(figsize=(10,10))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        gif = os.path.join(self.img_path, str(self.type) +'_dcgan_%d.gif'%(self.epoch))
        anim.save(gif, dpi=80,writer='imagemagick')       
 


if __name__ == "__main__":

    CelebA_G = Generator()
    CelebA_D = Discriminator()
    DCGAN_train = DCGAN_Trainer(CelebA_G,CelebA_D)
    
    dataloader = load_data(DCGAN_train.image_size, DCGAN_train.dataroot, DCGAN_train.batch_size)
    DCGAN_train.training(dataloader) 

    # rect_G = My_G()
    # rect_D = My_D()

    # My_DCGAN = DCGAN_Trainer(rect_G, rect_D)
    # My_DCGAN.epoch = 1000
    # My_DCGAN.lr = 0.00001
    # My_DCGAN.image_size = 0
    # My_DCGAN.batch_size = 8
    # My_DCGAN.dataroot = '/home/admin11/Data_test/test'
    # My_DCGAN.img_path = '/home/admin11/1.my_zone/out_imgs/1.DCGAN/My_rect'
    # My_DCGAN.model_path = '/home/admin11/1.my_zone/model/1.DCGAN/My_rect'
    # My_DCGAN.type = 'My_Rect'
    # dataloader = load_data(My_DCGAN.image_size, My_DCGAN.dataroot, My_DCGAN.batch_size)
    # My_DCGAN.training(dataloader) 
    
    
    
    

    
            



