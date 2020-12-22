# -*- coding: utf-8 -*-
'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-08 20:02:57
LastEditors: jinlong li
LastEditTime: 2020-10-05 21:49:00
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

# import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import numpy as np
import time
import os 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

from utils import log_gaussian, noise_sample, weight_init
from Mydataloader import load_data
# import tensorflow



'''Method 1 input　3*270*1280 imgs  '''

class FrontEnd_rectangle(nn.Module):
    ''' 
    shared frond part of D and Q 
    intput: 3*270*1280
    output:  bs*256*4*20
    num_conv: 6
    ''' 
    def __init__(self,):
        super(FrontEnd_rectangle, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (4,4), stride=2, padding=1), # 64, 640, 135
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3,3), stride=1, padding=1), #128, 640, 135
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        
        self.conv2 = nn.Sequential( 
            nn.Conv2d(128, 256, (3,3), stride=2, padding=1), #256, 320, 68
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), #256, 160, 34
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, (3,3), stride=1, padding=1), #512, 160, 34
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), #512, 80, 17
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, (2,1), stride=1, padding=0), #512, 80, 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), # 512, 40, 8
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 256, (3,3), stride=1, padding=1), #256, 40, 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.AvgPool2d(2, stride=2), # 256, 20, 4
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Info_D_rectangle(nn.Module):
    '''
    input: 256*4*20
    output: bs* 10
    num_conv: 1
    '''
    def __init__(self):
        super(Info_D_rectangle, self).__init__()

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, (2,3), stride=1, padding=1), # 128, 20, 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), # 128, 10, 2  
        )

        self.fc = nn.Sequential(
            nn.Linear(128*2*10, 1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024, 12),
            nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Q_rectangle (nn.Module):
    '''
    input: 256*4*20
    output: bs*12, bs*2, bs*2
    num_conv: 1
    '''
    def __init__(self):
        super(Q_rectangle, self).__init__()
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, (4,4), stride=2, padding=1), # 128, 10, 2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), # 128, 5, 1
            nn.Conv2d(128,64, (1,1), 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True) #64, 5, 1  
        )
        
        self.conv_disc = nn.Conv2d(64,12, (1,5), stride=1, padding=0)  # 12, 1, 1
        self.conv_mu = nn.Conv2d(64, 2,  (1,5), stride=1, padding=0)   # 2, 1, 1
        self.conv_var = nn.Conv2d(64, 2,  (1,5), stride=1, padding=0)  # 2, 1, 1

    def forward(self, x):
        y = self.conv6(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var 

class Info_G_rectangle(nn.Module):
    '''
    input: bs*94(80+12+2)*1*1
    output: bs* 3*270*1280
    num_conv: 11
    '''
    def __init__(self):
        super(Info_G_rectangle, self).__init__()
        #input: (80+12+2)*1*1 
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(94, 512, (2,4), stride=1, padding=0,bias=False), # 512*4*2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (1,3), 1, 0, bias=False), # 512, 6, 2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)            
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (2, 3), 1, 0, bias=False), #512, 8, 3
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (2,3), 1, 0, bias=False), # 256, 10, 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (1,4), 2, 1, bias=False), # 256, 20, 5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)           
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (3,4), 2, 1, bias=False), # 256, 40, 9
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)  
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (3,4), 2, 1, bias=False), # 256, 80, 17
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)  
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (2,2), 2, 0, bias=False), # 128, 160, 34
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)  
        )
        self.layerdeep = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4,4), 2, 1, bias=False), # 64, 320, 68
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 16, (3,4), 2, 1, bias=False), # 16, 640, 135
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,3, (2,2), 2, 0, bias=False), # 3, 1280, 270
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layerdeep(x)
        return x


''' Method 2  input 3*512*512 pixels'''

class FrontEnd_square(nn.Module):
    ''' 
    shared frond part of D and Q 
    intput: 3*512*512
    output:  bs*64*8*8
    num_conv: 6
    ''' 
    def __init__(self):
        super(FrontEnd_square, self).__init__()

        self.conv1 = nn.Sequential(
            # inout 3*512*512
            nn.Conv2d(3, 64, 5, stride=2, padding=2),  # batch, 64, 256, 256
            nn.LeakyReLU(0.2, True),
            )
        self.conv2 = nn.Sequential(
            # inout 64*256*256
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # batch, 128, 256, 256
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 128, 128, 128
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256, 3, stride=1, padding=1),  # batch, 256, 128, 128
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 256, 64, 64
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,256, 3, stride=1, padding=1),  # batch, 256, 64, 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 256, 32, 32
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256,128, 3, stride=1, padding=1),  # batch, 128, 32, 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 128, 16, 16
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,64, 3, stride=1, padding=1),  # batch, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 8, 8
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class Info_D_square(nn.Module):
    '''
    input: 64*8*8
    output: bs* 4
    num_conv: 1
    '''
    def __init__(self):
        super(Info_D_square, self).__init__()

        self.conv7 = nn.Sequential(
            nn.Conv2d(64,32, 3, stride=1, padding=1),  # batch, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 32, 4, 4
        )
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 4),
            nn.LeakyReLU(0.2, True),
            # nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Q_square (nn.Module):
    '''
    input: 64*8*8
    output: bs*4, bs*2, bs*2
    num_conv: 1
    '''
    def __init__(self):
        super(Q_square, self).__init__()
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(64,32, 3, stride=1, padding=1),  # batch, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 4, 4
            nn.Conv2d(32, 16, 1,1, 0), # 16, 4, 4
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
        )
        
        self.conv_disc = nn.Conv2d(16, 4, 4, stride=1, padding=0)  # 4, 1, 1
        self.conv_mu = nn.Conv2d(16, 2,  4, stride=1, padding=0)   # 2, 1, 1
        self.conv_var = nn.Conv2d(16, 2,  4, stride=1, padding=0)  # 2, 1, 1

    def forward(self, x):
        y = self.conv7(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var 


class Info_G_square(nn.Module):
    '''
    input: bs*86(80+4+2)*1*1
    output: bs* 3*512*512
    num_conv: 11
    '''
    def __init__(self):
        super(Info_G_square, self).__init__()
        #input: (80+4+2)*1*1 
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(86, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸256x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(256)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(128)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸(128)x64x64
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # layer6输出尺寸(128)x128x128
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # layer7输出尺寸(128)x256x256
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # layer8输出尺寸3x512x512
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out

''' Method 3 input 3*1280*1280  pixels'''

class My_D(nn.Module):
    ''' 
    shared frond part of D and Q 
    intput: 3*1280*1280
    output:  (ndf*16)*10*10
    num_conv: 4
    ''' 
    def __init__(self):
        super(My_D, self).__init__()
        # self.ngpu = ngpu
        self.ndf = 64
        self.nc = 3
        self.main = nn.Sequential(
            # input is (nc) x 1280 x 1280
            nn.Conv2d(self.nc, self.ndf*2, 4, 2, 1, bias=False), # (ndf*2) x 640 x 640
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf*2, self.ndf *4, 4, 2, 1, bias=False), # (ndf*4) x 160 x 160
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False), # (ndf*8) x 40 x 40
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf *8, self.ndf* 16, 4, 2, 1, bias=False),# (ndf*16) x 10 x 10
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 2, 0, bias=False),# (ndf*16) x 4 x 4
            # nn.BatchNorm2d(self.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 1, 0, bias=False),# (ndf*16) x 1 x 1
            # # nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        # out = x.view(x.size(0), -1)
        # out = self.fc(out) 
        return x


class Info_D_square1280(nn.Module):
    '''
    input: (ndf*16)*10*10
    output: (ndf*16) x 36
    num_conv: 2
    '''
    def __init__(self):
        super(Info_D_square1280, self).__init__()
        self.ndf = 64
        self.nc = 3
        self.main = nn.Sequential(
            nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 2, 0, bias=False),# (ndf*16) x 4 x 4
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 1, 0, bias=False),# (ndf*16) x 1 x 1
        )
        self.fc = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.ndf* 16, 36),
            nn.LeakyReLU(0.2, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Q_square1280(nn.Module):
    '''
    input:(ndf*16)*10*10
    output: bs*4, bs*2, bs*2
    num_conv: 1
    '''
    def __init__(self):
        super(Q_square1280, self).__init__()
        self.ndf = 64
        self.nc = 3

        self.main = nn.Sequential(
            nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 2, 0, bias=False),# (ndf*16) x 4 x 4
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 1, 0, bias=False),# (ndf*16) x 1 x 1
        )
        self.conv_disc = nn.Conv2d(self.ndf *16, 36*4, 4, stride=1, padding=0)  # 4, 1, 1
        self.conv_mu = nn.Conv2d(self.ndf *16, 4,  4, stride=1, padding=0)   # 2, 1, 1
        self.conv_var = nn.Conv2d(self.ndf *16, 4,  4, stride=1, padding=0)  # 2, 1, 1


    def forward(self, x):
        y = self.main(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var 



class My_G(nn.Module):
    '''
    input: bs*86(80+4+2)*1*1
    output: bs* 3*1280*1280
    num_conv: 9
    '''
    def __init__(self, nz):
        super(My_G, self).__init__()
        # self.ngpu = ngpu
        self.nz = nz
        self.ngf = 128
        self.nc = 3

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
            
    def forward(self, input):
        return self.main(input)







''' InfoGAN training'''


class InfoGAN_Trainer:
    def __init__(self, Generation, FrontEnd, Discriminator, Q):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Generation = Generation
        self.FrontEnd = FrontEnd
        self.Discriminator = Discriminator
        self.Q = Q

        # self.ngf = 64 # generate channel
        # self.ndf = 64 # discriminative channel
        self.resize = 0
        self.batch_size = 4  # >1
        self.dis_c = 4
        self.con_c = 4
        # self.label = 10
        self.n_z = 80
        self.dis_c_dim = 36
        self.beta = 0.5
        self.lr_D = 0.00001
        self.lr_G = 0.0001
        self.epoch = 200

        self.img_path = '/home/admin11/1.my_zone/out_imgs/3.infoGAN'
        self.model_path = '/home/admin11/1.my_zone/model/3.infoGAN'
        self.type = 'test_2'
        self.log = 'InfoGAN_log'       
        

        self.G_loss = []
        self.D_loss = []
        self.img_list = []

        
    

    def training(self, img_path):

        train_data = load_data(self.resize,img_path,self.batch_size)

        # real_img = torch.FloatTensor(self.batch_size, self.channel, self.img_length, self.img_width).to(self.device)
        label = torch.FloatTensor(self.batch_size, self.dis_c_dim).to(self.device)

        # real_img = Variable(real_img)
        label = Variable(label)
        self.Generation.apply(weight_init)
        self.FrontEnd.apply(weight_init)
        self.Discriminator.apply(weight_init)
        self.Q.apply(weight_init)

        self.Generation.to(self.device)
        self.FrontEnd.to(self.device)
        self.Discriminator.to(self.device)
        self.Q.to(self.device)
        fw = open(self.log +'_' + self.type +'.txt', 'w')



        criterionD = nn.BCELoss().to(self.device)
        criterionQ_dis = nn.CrossEntropyLoss().to(self.device)
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params':self.FrontEnd.parameters()}, {'params':self.Discriminator.parameters()}], lr=self.lr_D, betas=(self.beta, 0.999))
        optimG = optim.Adam([{'params':self.Generation.parameters()}, {'params':self.Q.parameters()}], lr=self.lr_G, betas=(self.beta, 0.999))

        if not os.path.exists(self.img_path+'/' + self.type):
            os.makedirs(self.img_path+ '/' +self.type)
        if not os.path.exists(self.model_path+ '/'+self.type):
            os.makedirs(self.model_path+ '/'+self.type)
        # if not os.path.exists(self.save_path+'/InfoGAN_log'):
        #     os.makedirs(self.save_path+'InfoGAN_log')
        writer = SummaryWriter(os.path.join(self.img_path, self.log))

        print("Starting Training Loop...")

        start_time = time.time()
        iters = 0

        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for i, (data, _) in enumerate(train_data, 0):

                bs = data.size(0)
                real_img = data.to(self.device)
                iters +=1

                # Updating discriminator
                optimD.zero_grad()
                # label = torch.full((b_size, ), real_label, device=device)
                # real part
                label.data.resize_(bs, self.dis_c_dim)
                fe_out1 = self.FrontEnd(real_img)
                probs_real = self.Discriminator(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # pdb.set_trace()
                # fake part
                z, idx = noise_sample(self.dis_c, self.dis_c_dim, self.con_c, self.n_z, self.batch_size, self.device)
                fake_img = self.Generation(z)
                fe_out2 = self.FrontEnd(fake_img.detach())
                probs_fake = self.Discriminator(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake
                optimD.step()

                # Updating Generator and Q
                optimG.zero_grad()
                fe_out = self.FrontEnd(fake_img)
                probs_fake = self.Discriminator(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)
                # pdb.set_trace()
                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).to(self.device)
                target = Variable(class_)


                dis_loss = 0
                for j in range(int(self.dis_c)):
                    # pdb.set_trace()
                    dis_loss += criterionQ_dis(q_logits[:,j*36: j*36 +36], target[j])

                    # an = Variable(torch.unsqueeze(q_logits[j,:],0))
                    # an = Variable(torch.unsqueeze(q_logits[:, j], 0))

                    # dis_loss += criterionQ_dis(q_logits[:, j], target[:,j])
                    


                # dis_loss = criterionQ_dis(q_logits, target)
                # pdb.set_trace()
                con_c = z[:,self.n_z+self.dis_c*self.dis_c_dim:].squeeze()
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()
                # Save Losses for plotting later
                self.G_loss.append(G_loss.item())
                self.D_loss.append(D_loss.item())


                if i%50 == 0:

                    print('InfoGAN_%s:\tEpoch/Iter:%d/%d/{%d}, Dloss: %.4f, Gloss: %4f'%(
                        self.type, self.epoch, epoch+1, i+1, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )
                    fw.write('epoch ' + str(self.epoch+1) + ': ' + str(epoch+1)+'; ' + str(len(train_data)) +': '+ str(i+1) + ' '+ 'Loss_D: '+ str(D_loss.data.cpu().numpy())+' Loss_G: '+ str(G_loss.data.cpu().numpy()))
                    fw.write('\n')

                    writer.add_scalar('Train_InfoGAN/D_Loss',D_loss.data.cpu().numpy(), iters )
                    writer.add_scalar('Train_InfoGAN/G_Loss',G_loss.data.cpu().numpy(), iters )
                

            epoch_time = time.time() - epoch_start_time
            print("InfoGAN_%s: Time taken for Epoch %d: %.2fs" %(self.type, epoch + 1, epoch_time))
            # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
            gen_data = self.Generation(z).detach().cpu()
            self.img_list.append(vutils.make_grid(gen_data, nrow=4, padding=2, normalize=True))
            # Generate image to check performance of generator.
            if(epoch+1)%2 == 0:
                with torch.no_grad():
                    gen_data = self.Generation(z).detach().cpu()
                    save_image(gen_data, os.path.join(os.path.join(self.img_path, self.type), str(self.type)+'_infogan_epoch_%d.png'%(epoch+1)), normalize=True)
            writer.add_image('generation_images', fake_img[0].detach().detach())

            #save the net weights
            if ((epoch+1)%50 == 0):
                torch.save({
                    'netG' : self.Generation.state_dict(),
                    'FrontEnd' : self.FrontEnd.state_dict(),
                    'netD' : self.Discriminator.state_dict(),
                    'netQ' : self.Q.state_dict(),
                    'optimD' : optimD.state_dict(),
                    'optimG' : optimG.state_dict(),
                    }, os.path.join(os.path.join(self.model_path, self.type), str(self.type)+'_infoGAN_%d.pth'%(epoch+1)))
  

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_loss,label="G")
        plt.plot(self.D_loss,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.model_path, self.type), str(self.type)+ '_infoGAN_epoch_%d.png'% (epoch+1)))

        # Animation showing the improvements of the generator.
        # fig = plt.figure(figsize=(10,10))
        # plt.axis("off")
        # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        # anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        # gif = os.path.join(self.model_path, str(self.type) +'_infoGAN_%d.gif'%(self.epoch))
        # anim.save(gif, dpi=80,writer='imagemagick')
        # anim.save(gif, writer='pillow')



if __name__ == "__main__":

    img_path_square= '/home/admin11/Data_test/Dataset/8.MinImg_training'
    nz = 228 #228
    G_square = My_G(nz)
    D_square = Info_D_square1280()
    FE_square = My_D()
    Q_square = Q_square1280()

    Train_square = InfoGAN_Trainer(G_square, FE_square, D_square, Q_square)
    Train_square.resize = 0
    Train_square.type = 'test_2'
    # Train_square.dis_c_dim = 4
    Train_square.training(img_path_square)
    
                


            
                

                


                 



        


    








        
                

        
        
    






    



    
