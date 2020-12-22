'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-09 15:52:53
LastEditors: jinlong li
LastEditTime: 2020-09-24 18:28:58
'''

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pdb



# 加载数据集
def load_data(resize,data,batch_size, num_workers):
    if int(resize) > 0:
        # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
        data_tf = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    else:
        data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        

    train_dataset =datasets.ImageFolder(data, transform=data_tf)# data 为自己数据集的目录
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_workers)

    return train_loader


        # Create the dataset
        # dataset = datasets.ImageFolder(self.dataroot,
        #                         transform=transforms.Compose([
        #                             transforms.Resize(self.image_size),
        #                             transforms.CenterCrop(self.image_size),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                         ]))
        # # Create the dataloader
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)


class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    
    """
    Sample random noise vector for training.
    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim:  Dimension of discrete latent code.(generally dis_c_dim = numbers of labels)
    n_con_c : Number of continuous latent code.
    n_z : Dimension of noise.
    batch_size : Batch Size
    device : GPU/CPU
    """


    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    # pdb.set_trace()
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
        
        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


# custom weights initialization called on netG and netD
def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)



