'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-11 13:19:17
LastEditors: jinlong li
LastEditTime: 2020-09-24 10:49:22
'''
from torchvision import get_image_backend
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
import os
from torchvision.utils import save_image
import cv2
import PIL
import time
import threading

from Mydataloader import load_data
# from InfoGAN import Generation


''' Method 1 CGAN based on Conv, input 3*1280*270 images'''

class Con_D_rectangle(nn.Module):
    ''' 
    intput: 3*1280*270 pixels
    output: 12 classes
    num_Conv: 8
    ''' 
    def __init__(self):
        super(Con_D_rectangle, self).__init__()

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
            nn.Conv2d(512, 256, (2,1), stride=1, padding=0), #256, 80, 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), # 256, 40, 8
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, (3,3), stride=1, padding=1), #128, 40, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.AvgPool2d(2, stride=2), # 128, 19, 4
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 32, (2,3), stride=1, padding=1), # 32, 20, 4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2), # 32, 10, 2  
        )

        self.fc = nn.Sequential(
            nn.Linear(32*2*10, 1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 12),
            nn.LeakyReLU(0.2, True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Con_G_rectangle(nn.Module):
    '''
    intput: 100(88+12)*1*1
    output: 3*1280*270
    num_Conv: 11
    '''
    def __init__(self):
        super(Con_G_rectangle, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, (2,4), stride=1, padding=0,bias=False), # 512*4*2
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



''' Method 2 CGAN based on Conv, input:  3*512*512 images '''

class Con_D_square(nn.Module):
    '''
    input: 3*512*512 piels
    output: 12 classes
    num_layers: 8

    '''
    def __init__(self):
        super(Con_D_square, self).__init__()
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
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)     
        # z = self.feature(x)
        # self.feature_num = z.cpu()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Con_G_square(nn.Module):
    def __init__(self):
        '''
        input: 92(88+4)*1*1
        output: 3*512*512
        num_Conv:8
        '''
        super(Con_G_square, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸512x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(92, 512, kernel_size=4, stride=1, padding=0, bias=False),
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

''' CGAN Training '''
class CGAN_Trainer:
    def __init__(self, G, D):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = G
        self.D = D
        self.batch_size = 4
        self.resize = 512
        self.epoch = 50
        # Size of z latent vector (i.e. size of generator input)
        self.nz = 88
        # Learning rate for optimizers
        self.lr = 0.0001
        self.num_label = 12
        self.beta = 0.5
        self.G_losses = []
        self.D_losses = []
        
        self.save_path = '/home/admin11/1.my_zone/out_imgs/2.CGAN'
        self.model_path = '/home/admin11/1.my_zone/model/CGAN'
        self.type ='1.square'
        

    def training(self, img_path):

        train_data = load_data(self.resize,img_path,self.batch_size)
        self.G.to(self.device) # generator model
        self.D.to(self.device) # discriminator model

        criterion = nn.BCELoss()  # binary cross entropy

        optimizerG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta, 0.999))
        optimizerD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta, 0.999))

        start_time = time.time()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        print("Starting Training Loop...")
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for i, data in enumerate(train_data, 0):

                # 固定生成器G，训练鉴别器D
                optimizerD.zero_grad()
                ## 让D尽可能的把真图片判别为1
                real_img = data[0].to(self.device)
                #data[images,label]  data[1]是标签信息
                labels_onehot = np.zeros((self.batch_size, self.num_label))
                labels_onehot[np.arange(self.batch_size),data[1].numpy()]=1
                real_label=Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
                # compute loss of real_img
                real_out = self.D(real_img)
                # Calculate loss on all-real batch
                errD_real = criterion(real_out, real_label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = real_out.mean().item() # item是得到一个元素张量里面的元素值

                # 让D尽可能把假图片判别为0
                # Generate batch of latent vectors
                noise = torch.randn(real_img.size(0), self.nz, 1, 1) #随机生成向量
                labels_onehot = np.zeros((self.batch_size, self.num_label, 1, 1))
                labels_onehot[np.arange(self.batch_size),data[1].numpy()]=1
                noise=np.concatenate((noise.numpy(),labels_onehot),axis=1)
                noise=Variable(torch.from_numpy(noise).float()).to(self.device)
                
                # Generate fake image batch with G
                fake_img = self.G(noise)
                fake_label = Variable(torch.zeros(real_img.size(0),self.num_label)).to(self.device)#返回一个全为0的维度为real_img.size(0)的张量   这个有多形式
                
                # Classify all fake batch with D
                fake_out = self.D(fake_img.detach()) #避免梯度传到G，因为G不用更新
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(fake_out, fake_label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = fake_out.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                # 固定鉴别器D，训练生成器G
                optimizerG.zero_grad()    #netG.zero_grad() 有两种形式
                noise = torch.randn(real_img.size(0), self.nz, 1, 1) #
                noise=np.concatenate((noise.numpy(),labels_onehot),axis=1)
                noise=Variable(torch.from_numpy(noise).float()).to(self.device)
                # Generate fake image batch with G
                fake_img = self.G(noise)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.D(fake_img) 
                # Calculate G's loss based on this output
                errG = criterion(output, real_label)#得到假的图片与真实标签的loss
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()

                # Update G
                optimizerG.step()

                # Output training stats
                if i%10  == 0:
                    print('CGAN_%s: [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (self.type, (epoch+1), self.epoch, i, len(train_data),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
            epoch_time = time.time() - epoch_start_time
            print("CGAN_%s: Time taken for Epoch %d: %.2fs" %(self.type, epoch + 1, epoch_time))
            # Check how the generator is doing by saving G's output
            if epoch%5 == 0:
                with torch.no_grad():
                    gen_data = self.G(noise).detach().cpu()
                    save_image(gen_data.data, os.path.join(self.save_path, str(self.type)+'cgan_epoch_%d.png' % (epoch+1)), normalize=True)

        #save the net
        state_1 = {'net':self.G.state_dict(), 'optimizer':optimizerG.state_dict(), 'epoch':epoch}
        torch.save(state_1, os.path.join(self.model_path,str(self.type)+'cgan_%d.pth' %(epoch+1)))
        state_2 = {'net':self.D.state_dict(), 'optimizer':optimizerD.state_dict(), 'epoch':epoch}
        torch.save(state_2, os.path.join(self.model_path, str(self.type)+'cgan_%d.pth'%(epoch+1)))

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.model_path, str(self.type)+'loss_cgan_epoch_%d.png'% (epoch+1)))



if __name__ == "__main__":
    
    img_path_rectangle= '/home/admin11/Data_test/Dataset/2.MinImg_training'

    G_rectangle = Con_G_rectangle() 
    D_rectangle = Con_D_rectangle()

    Train_rectangle = CGAN_Trainer(G_rectangle, D_rectangle)
    Train_rectangle.resize = 0
    Train_rectangle.type = 'rectangle'
    # Train_rectangle.training(img_path)
    T1 = threading.Thread(target=Train_rectangle.training, args=(img_path_rectangle,))



    img_path_square= '/home/admin11/Data_test/Dataset/3.MAXImg_training'

    G_square = Con_G_square()
    D_square = Con_D_square()

    Train_square = CGAN_Trainer(G_square, D_square)
    Train_square.num_label = 4
    # Train_square.training(img_path)
    T2 = threading.Thread(target=Train_square.training, args=(img_path_square,))



    T1.start()
    T2.start()
    T1.join()
    T2.join()




    
    

    
    
            

            
         


