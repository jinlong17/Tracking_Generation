# -*- coding: utf-8 -*-
'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-30 14:36:16
LastEditors: jinlong li
LastEditTime: 2020-09-24 11:48:40
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
from torch.autograd import Variable

from utils import weight_init, load_data
from GAN import My_G

''' 2.G_myRectangle:'''
class My_D(nn.Module):
    def __init__(self):
        super(My_D, self).__init__()
        # self.ngpu = ngpu
        self.ndf = 64
        self.nc = 3
        # self.main = nn.Sequential(
        #     # input is (nc) x 1280 x 1280
        #     nn.Conv2d(self.nc, self.ndf, (4,4), 2, 1, bias=False), # (ndf) x 640 x 640
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf, self.ndf * 2, (4,4), 2, 1, bias=False), #(ndf*2) x 320 x 320
        #     nn.BatchNorm2d(self.ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 2, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 160 x 160
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 80 x 80
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 40 x 40
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 4, (4,4), 2, 1, bias=False), # (ndf*4) x 20 x 20
        #     nn.BatchNorm2d(self.ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 8, (4,4), 2, 1, bias=False), # (ndf*4) x 10 x 10
        #     nn.BatchNorm2d(self.ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 8, self.ndf * 16, (4,4), 2, 1, bias=False), # (ndf*16) x 5 x 5
        #     nn.BatchNorm2d(self.ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 16, 36, (5,5), 1, 0, bias=False), # 36 x 1 x 1
        # )
        self.fc = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.ndf* 16, 36),
            nn.LeakyReLU(0.2, True),
            nn.Sigmoid()
        )
        # self.main = nn.Sequential(
        #     # input is (nc) x 1280 x 1280
        #     nn.Conv2d(self.nc, self.ndf*2, 6, 4, 1, bias=False), # (ndf*2) x 320 x 320
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf*2, self.ndf *4, 6, 4, 1, bias=False), # (ndf*4) x 80 x 80
        #     nn.BatchNorm2d(self.ndf*4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf * 4, self.ndf * 8, 6, 4, 1, bias=False), # (ndf*8) x 20 x 20
        #     nn.BatchNorm2d(self.ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf *8, self.ndf* 16, 4, 2, 1, bias=False),# (ndf*16) x 10 x 10
        #     nn.BatchNorm2d(self.ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 2, 0, bias=False),# (ndf*16) x 4 x 4
        #     nn.BatchNorm2d(self.ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 1, 0, bias=False),# (ndf*16) x 1 x 1
        #     # nn.Sigmoid()
        # )
        self.main = nn.Sequential(
            # input is (nc) x 1280 x 1280
            nn.Conv2d(self.nc, self.ndf*2, 4, 2, 1, bias=False), # (ndf*2) x 320 x 320
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf*2, self.ndf *4, 4, 2, 1, bias=False), # (ndf*4) x 80 x 80
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False), # (ndf*8) x 20 x 20
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(self.ndf *8, self.ndf* 16, 4, 2, 1, bias=False),# (ndf*16) x 10 x 10
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 2, 0, bias=False),# (ndf*16) x 4 x 4
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf *16, self.ndf* 16, 4, 1, 0, bias=False),# (ndf*16) x 1 x 1
            # nn.Sigmoid()
        )
    # def forward(self, input):
    #     return self.main(input)


    def forward(self, input):
        x = self.main(input)
        out = x.view(x.size(0), -1)
        out = self.fc(out) 
        return out




'''
CGAN training
'''
class CGAN_Trainer():
    def __init__(self, G, D):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = G
        self.D = D
        self.loss = []
        self.image_size = 0
        self.batch_size = 2
        self.worker = 8
        self.dataroot = '/home/admin11/Data_test/Dataset/8.MinImg_training'
        # self.dataroot = '/home/admin11/Data_test/test'
        self.D_lr = 0.00001
        self.G_lr = 0.0001
        self.epoch = 200
        self.beta1 = 0.5
        self.nz = 84
        self.num_label = 36


        self.img_list = []
        self.G_loss = []
        self.D_loss = []
        

        self.img_path = '/home/admin11/1.my_zone/out_imgs/2.CGAN/My_rect/imgs'
        self.model_path = '/home/admin11/1.my_zone/model/2.CGAN/My_rect'
        self.type ='test_8'
        self.log = 'CGAN_log'

    def training(self, dataloader):
        self.G.to(self.device)
        self.D.to(self.device)
        self.G.apply(weight_init)
        self.D.apply(weight_init)
        fw = open(os.path.join(os.path.join(self.model_path, self.type), self.log +'.txt'), 'w')

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(self.batch_size, self.nz+self.num_label, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(self.beta1, 0.999))

        if not os.path.exists(self.img_path+'/' + self.type):
            os.makedirs(self.img_path+ '/' +self.type)
        if not os.path.exists(self.model_path+ '/'+self.type):
            os.makedirs(self.model_path+ '/'+self.type)
        # if not os.path.exists(self.img_path+ self.log):
        #     os.makedirs(self.img_path+self.log)
        writer = SummaryWriter(os.path.join(self.img_path, self.log))

        iters = 0

        print("Starting Training Loop...")
        for epoch in range(self.epoch):

            for i, (data, label_ori) in enumerate(dataloader, 0):

                iters +=1

                self.D.zero_grad()
                real_img = data.to(self.device)
                bs = real_img.size(0)

                labels_onehot = np.zeros((self.batch_size, self.num_label))
                labels_onehot[np.arange(self.batch_size),label_ori.numpy()]=1
                real_label=Variable(torch.from_numpy(labels_onehot).float()).cuda()
                
                # label = torch.full((self.batch_size,), real_label, device=self.device)
                # Forward pass real batch through D
                output = self.D(real_img)
                # Calculate loss on all-real batch
                errD_real = criterion(output, real_label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(self.batch_size, self.nz, 1, 1)
                labels_onehot = np.zeros((self.batch_size, self.num_label, 1, 1))
                labels_onehot[np.arange(self.batch_size), label_ori.numpy()]=1
                noise=np.concatenate((noise.numpy(),labels_onehot),axis=1)
                noise=Variable(torch.from_numpy(noise).float()).to(self.device)
                
                # Generate fake image batch with G
                fake = self.G(noise)
                fake_label = Variable(torch.zeros(real_img.size(0),self.num_label)).to(self.device)
                # label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, fake_label)
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

                output = self.D(fake)
                # Calculate G's loss based on this output
                errG = criterion(output, real_label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('CGAN training: [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.epoch, i+1, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    fw.write('epoch ' + str(self.epoch+1) + ': ' + str(epoch+1)+'; ' + str(len(dataloader)) +': '+ str(i+1) + ' '+ 'Loss_D: '+ str(errD.item())+' Loss_G: '+ str(errG.item()))
                    fw.write('\n')

                # Save Losses for plotting later
                self.G_loss.append(errG.item())
                self.D_loss.append(errD.item())
                writer.add_scalar('D_Loss',errD.item(), iters)
                writer.add_scalar('G_Loss',errG.item(), iters)

                # Check how the generator is doing by saving G's output on fixed_noise
                if (i == len(dataloader)-1):
                    with torch.no_grad():
                        fake_img = self.G(noise).detach().cpu()
                        save_image(fake, os.path.join(os.path.join(self.img_path, self.type), str(self.type)+'_dcgan_%d.png'%(epoch+1)), normalize=True)
                        writer.add_image('generation_images', fake_img[0])
                    self.img_list.append(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True))

            #save the net
            if ((epoch+1)%50 == 0):
                state_1 = {'net':self.G.state_dict(), 'optimizer':optimizerG.state_dict(), 'epoch':epoch}
                torch.save(state_1, os.path.join(os.path.join(self.model_path, self.type),str(self.type)+'_G_cgan_%d.pth' %(epoch+1)))
                state_2 = {'net':self.D.state_dict(), 'optimizer':optimizerD.state_dict(), 'epoch':epoch}
                torch.save(state_2, os.path.join(os.path.join(self.model_path, self.type), str(self.type)+'_D_cgan_%d.pth'%(epoch+1)))

        plt.figure(figsize=(10,5))
        plt.title(str(self.type) + "_CGAN Loss During Training")
        plt.plot(self.D_loss, label="D_Loss")
        plt.plot(self.G_loss, label="G_Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.model_path, self.type), str(self.type) + '_cgan_%03d_epoch.png'%(self.epoch)))

        # Animation showing the improvements of the generator.
        fig = plt.figure(figsize=(10,10))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        gif = os.path.join(self.img_path, str(self.type) +'_cgan_%d.gif'%(self.epoch))
        anim.save(gif, dpi=80,writer='imagemagick')

if __name__ == "__main__":

    nz = 120
    rect_G = My_G(nz)
    rect_D = My_D()

    My_CGAN = CGAN_Trainer(rect_G, rect_D)
    # My_CGAN.batch_size = 2
    My_CGAN.image_size = 0

    dataloader = load_data(My_CGAN.image_size, My_CGAN.dataroot, My_CGAN.batch_size)

    My_CGAN.training(dataloader) 
