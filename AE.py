# -*- coding: utf-8 -*-
'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-24 16:35:52
LastEditors: jinlong li
LastEditTime: 2020-08-09 10:13:19
'''
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

from utils import load_data
import pdb
 
# from 1.Conv_VAE import Flatten, Unflatten
class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)
'''
AE based on Conv,  images inout: 3*96*96
'''

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.nc = 3
        self.ndf = 64
        
        self.encoder = nn.Sequential(
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
            nn.Conv2d(self.ndf * 8, 100, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            nn.ReLU(True)
        )

    def forward(self, x):
        output = self.encoder(x)
        return output
        
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        
        self.ngf = 64
        self.nc = 3

        self.decoder = nn.Sequential(
            # input is 100 x 1 x 1
            nn.ConvTranspose2d(100, self.ngf * 8, 4, 1, 0, bias=False),
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
    def forward(self, encode):
        output = self.decoder(encode)
        return output



class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.nc = 3
        self.ndf = 64
        self.ngf = 64

        self.encoder = nn.Sequential(
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
            nn.Conv2d(self.ndf * 8, 100, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            # input is 100 x 1 x 1
            nn.ConvTranspose2d(100, self.ngf * 8, 4, 1, 0, bias=False),
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
        # self.encoder = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ndf * 2, 1.e-3),
        #     nn.LeakyReLU(0.2),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ndf * 4, 1.e-3),
        #     nn.LeakyReLU(0.2),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ndf * 8, 1.e-3),
        #     nn.LeakyReLU(0.2),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ndf * 8, 1.e-3),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 8, 100, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(100, 1.e-3),  
        #     nn.LeakyReLU(0.2)
        #     # nn.Sigmoid()
        #     # nn.ReLU(True)
        # )


        # self.decoder = nn.Sequential(
        #     # input is 100 x 1 x 1
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(100, self.ngf*8, 3, 1),
        #     nn.BatchNorm2d(self.ngf*8, 1.e-3),
        #     nn.LeakyReLU(0.2),
            
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(self.ngf*8, self.ngf*4, 3, 1),
        #     nn.BatchNorm2d(self.ngf*4, 1.e-3),
        #     nn.LeakyReLU(0.2),

        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(self.ngf*4, self.ngf*2, 3, 1),
        #     nn.BatchNorm2d(self.ngf*2, 1.e-3),
        #     nn.LeakyReLU(0.2),

        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(self.ngf*2, self.ngf, 3, 1),
        #     nn.BatchNorm2d(self.ngf, 1.e-3),
        #     nn.LeakyReLU(0.2),
        #     #add a layer
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(self.ngf, self.ngf, 3, 1),
        #     nn.BatchNorm2d(self.ngf, 1.e-3),
        #     nn.LeakyReLU(0.2),

        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(self.ngf, self.nc, 3, 1),

        #     nn.LeakyReLU(0.2),
        #     nn.ReLU(),
        #     nn.Sigmoid()

        # )


    def forward(self, x):
        encode = self.encoder(x)
        # vetor = self.flatten(encode)
        decode = self.decoder(encode)
        return encode, decode






class AE_Trainer:
    def __init__(self, decoder, encoder):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = AE
        self.model_D = decoder
        self.model_E = encoder
        self.loss = []
        self.image_size = 64
        self.batch_size = 128
        self.worker = 1
        self.lr = 0.0002
        self.epoch = 200
        self.weight = 10-5
        self.dataroot = '/home/admin11/Data_test/test_1000'#'/home/admin11/Data_test/celeba'
        self.img_path = '/home/admin11/1.my_zone/out_imgs/0.AE/CelebA/imgs'
        self.model_path = '/home/admin11/1.my_zone/model/0.AE/CelebA'
        self.type ='demo_1000'


    
    def training(self, dataloader):

        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        # criterion = nn.BCELoss()
        # optimizier = optim.Adam(self.model.parameters(), lr=self.lr)
        optimizier_D = optim.Adam(self.model_D.parameters(), lr=self.lr)
        optimizier_E = optim.Adam(self.model_E.parameters(), lr=self.lr)
        # self.model.to(self.device)
        self.model_D.to(self.device)
        self.model_E.to(self.device)

        if not os.path.exists(self.img_path+'/'+self.type):
            os.makedirs(self.img_path+'/'+self.type)
        if not os.path.exists(self.model_path+'/'+self.type):
            os.makedirs(self.model_path+'/'+self.type)
        # if not os.path.exists(self.img_path+'/AE_log'):
        #     os.makedirs(self.img_path+'/AE_log')
        writer = SummaryWriter(os.path.join(self.img_path, 'AE_log'))
        
        print("Starting Training Loop...")
        iters = 0
        for epoch in range(self.epoch):

            for i, (data, _) in enumerate(dataloader, 0):

                iters+=1

                bs = data.size(0)
                img = data.to(self.device)
                output = self.model_E(img)
                output = self.model_D(output)
                loss  = criterion(output, img)

                # optimizier.zero_grad()
                optimizier_D.zero_grad()
                optimizier_E.zero_grad()
                loss.backward()
                # optimizier.step()
                optimizier_D.step()
                optimizier_E.step()

                writer.add_scalar('AE_Loss',loss.data.float(), iters)
                self.loss.append(loss.data)

                
        
            writer.add_image('img', output.cpu().data[0])            
            # if (epoch+1) % 2 == 0:
            print("AE epoch: {}, loss is {}".format((epoch+1), loss.data))
            
            # pic = self.to_img(output.cpu().data)
            with torch.no_grad():
                save_image(output.cpu().data, os.path.join(os.path.join(self.img_path,self.type), str(self.type) + '_AE_{}.png'.format(epoch + 1)), normalize=True)

        torch.save(self.model_E, os.path.join(os.path.join(self.model_path, self.type), str(self.type) + '_E_{}.pth'.format(self.epoch)))
        torch.save(self.model_D, os.path.join(os.path.join(self.model_path, self.type), str(self.type) + '_D_{}.pth'.format(self.epoch)))

        plt.figure(figsize=(10,5))
        plt.title(str(self.type) + "_AE Loss During Training")
        plt.plot(self.loss, label="Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.model_path, self.type), str(self.type) + '_AE_%03depoch.png'%(self.epoch)))

if __name__ == "__main__":

    # CelebA_AE = AE()
    # AE_train = AE_Trainer(CelebA_AE)
    
    # dataloader = load_data(AE_train.image_size, AE_train.dataroot, AE_train.batch_size)
    # AE_train.training(dataloader)
    # 
    D = decoder()
    E = encoder()
    AE_train = AE_Trainer(D, E)
    
    dataloader = load_data(AE_train.image_size, AE_train.dataroot, AE_train.batch_size)
    AE_train.training(dataloader)  
    
    
                






        



        

        
        
        