#-*- coding: utf-8 -*-
'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-26 14:21:22
LastEditors: jinlong li
LastEditTime: 2020-08-11 18:13:52
'''

import numpy as np
import os
import pdb
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy
from torchvision import transforms, datasets

from utils import load_data
from AE import AE, encoder, decoder

from GAN import My_G, Generator

from PIL import Image
from numpy import average, dot, linalg
import matplotlib.image as mtimage






def build_path(path):
     if not os.path.exists(path):
         os.makedirs(path)



class Data_statistics:
    
    def __init__(self, img):

        # pdb.set_trace()

        self.dist = img
        self.dim = len(img[0])
        self.img_path = '/home/admin11/1.my_zone/test_imgs/AE_CelebA'
        self.type = 'CelebA'

        self.mean = []
        self.std = []
        self.median = []
        self.max = []
        self.ptp = []
        
        self.mean = [self.dist[i].mean(axis=0) for i in range(self.dim)]
        self.std = [np.std(self.dist[:, i]) for i in range(self.dim)]
        self.median = [np.median(self.dist[:, i]) for i in range(self.dim)]
        self.max = [np.max(self.dist[:, i]) for i in range(self.dim)]
        self.ptp = [np.ptp(self.dist[:, i]) for i in range(self.dim)]
        self.cov = numpy.cov(img, rowvar=False)
        

    ''' 1.histogram'''
    def drawing(self):


        plt.figure(figsize=(70,40))
        for index in range(1, self.dim+1):

            plt.subplot(2,5, index)
            plt.hist(self.dist[:, index-1], bins=50, color='steelblue')
            # plt.xlabel('density')
            # plt.ylabel('num')
            plt.title(str(self.type)+'_'+str(index))
        plt.savefig(os.path.join(self.img_path, '1.'+str(self.type)+ '_element_hist.png'))
        plt.show()

        # pdb.set_trace()
        x =[i+1 for i in range(self.dim)] 
        plt.figure(figsize=(70,30))

        plt.subplot(1, 5, 1)
        plt.plot(x, self.mean, '-.')
        plt.plot(x, self.mean, 's')
        plt.title(str(self.type)+'_Mean')
        plt.subplot(1, 5, 2)
        plt.plot(x, self.std, '-.')
        plt.plot(x, self.std, 's')
        plt.title(str(self.type)+'_Std')
        plt.subplot(1, 5, 3)
        plt.plot(x, self.median, '-.')
        plt.plot(x, self.median, 's')
        plt.title(str(self.type)+'_Median')
        plt.subplot(1, 5, 4)
        plt.plot(x, self.max, '-.')
        plt.plot(x, self.max, 's')
        plt.title(str(self.type)+'_Max')
        plt.subplot(1, 5, 5)
        plt.plot(x, self.ptp, '-.')
        plt.plot(x, self.ptp, 's')
        plt.title(str(self.type)+'_Ptp')
        plt.savefig(os.path.join(self.img_path, '2.'+str(self.type)+ '_statistics_hist.png'))
        plt.show()

    def cal(self, n):
        cov = self.cov
        mean = self.mean
        
        # for i in range(n):
        #     if i == 0:
        #         a = np.random.multivariate_normal(mean, cov, (n,1,1))
        #         # a = a.reshape(1, self.dim, 1, 1)
        #         a = torch.from_numpy(a)
        #         output = a.transpose(1,3)
        #         # output = torch.from_numpy(a)

        #     else:
        #         a = np.random.multivariate_normal(mean, cov, (n,1,1))
        #         a = torch.from_numpy(a)
        #         ten = a.transpose(1,3)
        #         # a = a.reshape(1, self.dim, 1, 1)
        #         # ten = torch.from_numpy(a)
        #         output = torch.cat([output,ten], 0)
        a = np.random.multivariate_normal(mean, cov, (n,1,1))
        a = torch.from_numpy(a)
        output = a.transpose(1,3)
        output = output.float()

        return output  

        

        
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    # image1 = get_thum(image1)
    # image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


def image_similarity(image1):
    
    img_aver = average(image1)
    norms = linalg.norm(img_aver, 2)

    return img_aver / norms



def img_match(targets, body):
    img_index = []
    img_max = []
    for i in range(len(targets)):
        value = image_similarity(targets[i])
        result = []
        for w, src in enumerate(body):
            result.append(dot(value, src))
        value_max = max(result)
        img_index.append(result.index(value_max))
        img_max.append(value_max)
        
    return img_index, img_max

def img_joint(img1, img2):
    if img1.size == img2.size:
        width, height = img1.size
        result = Image.new('L', (width * 3, height))
        result.paste(img1, box=(0,0))
        result.paste(img2, box=(width, 0))

    return result
    
       


if __name__ == "__main__":
    
    AE_E_path = '/home/admin11/1.my_zone/model/0.AE/CelebA/demo_1000/demo_1000_E_200.pth'
    AE_D_path = '/home/admin11/1.my_zone/model/0.AE/CelebA/demo_1000/demo_1000_D_200.pth'
    DCGAN_path = '/home/admin11/1.my_zone/model/1.DCGAN/CelebA/demo/demo_G_dcgan_500.pth'
    MyGAN_path = '/home/admin11/1.my_zone/model/1.DCGAN/CelebA/demo_new/demo_new_G_dcgan_500.pth'

    dataroot = '/home/admin11/Data_test/test_1000'
    test_path = '/home/admin11/1.my_zone/test_imgs'
    batch_size = 10
    nz = 100
    
    noise = torch.randn(batch_size, nz, 1, 1).cuda()

    # pdb.set_trace()

    test_path = '/home/admin11/1.my_zone/test_imgs'
    types = 'demo_1000'
    img_dis = np.loadtxt(os.path.join(test_path, str(types)+'_AE.txt'))
    Cal_img = Data_statistics(img_dis)
    fixed_noise = Cal_img.cal(batch_size)
    other_noise = fixed_noise.cuda()

    
    AE_model = torch.load(AE_D_path)
    ori_GAN = Generator()
    My_GAN = Generator()
    ori_GAN.load_state_dict(torch.load(DCGAN_path)['net'])
    My_GAN.load_state_dict(torch.load(MyGAN_path)['net'])

    AE_model.cuda()
    ori_GAN.cuda()
    My_GAN.cuda()

    D_imgs = AE_model(noise)
    ori_imgs = ori_GAN(noise)
    new_imgs = My_GAN(other_noise)

    imgs_list = os.listdir(os.path.join(dataroot, str(1)))
    trans_tensor = transforms.ToTensor()
    trans_nor = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    
    matching = []
    for i, img in enumerate(imgs_list):

        src = mtimage.imread(os.path.join(dataroot, str(1)+'/'+img))
        
        # src = trans_tensor(img)
        # src = trans_nor(src)
        matching.append(image_similarity(src))
    for i in range(batch_size):
        src1 = ori_imgs[i].mul_(255).add_(0.5).clamp_(0, 255).cpu().numpy().transpose((0, 2, 3, 1))
        src2 = new_imgs[i].mul_(255).add_(0.5).clamp_(0, 255).cpu().numpy().transpose((0, 2, 3, 1))

        ori_imgs[i] = ori_imgs[i]
        new_imgsp[i] = ori_imgs[i]

    ori_index, ori_max = img_match(ori_imgs, matching)
    new_index, new_max = img_match(new_imgs, matching)
    
    for i in range(batch_size):
        ori_result = img_joint(ori_imgs[i], mtimage.open(imgs_list[ori_index[i]]))
        new_result = img_joint(new_imgs[i], mtimage.open(imgs_list[new_index[i]]))
        
        ori_result = Image.fromarray((ori_result * 255).astype(np.uint8))
        new_result = Image.fromarray((new_result * 255).astype(np.uint8))
        
        ori_result.save(os.path.join(test_path, 'ori_img' + str(ori_max)+'.jpg'))
        new_result.save(os.path.join(test_path, 'new_img' + str(new_max)+'.jpg'))

    


        


        
            
            
            


        



    































    
    
    # # dataroot = '/home/admin11/Data_test/celeba'
    # dataroot = '/home/admin11/Data_test/test_1000'
    # # test_path = '/home/admin11/1.my_zone/test_imgs'
    # test_path = '/home/admin11/1.my_zone/test_imgs'
    # types = 'demo_1000'

    # # test_AE = encoder().cuda()

    # # test_AE = torch.load(model_path).cuda()
        
    # # build_path(os.path.join(os.path.join(test_path, types), types))
    # # dataset = load_data(64, dataroot, 20)

    # for i, (data, _) in enumerate(dataset, 0):

    #     # pdb.set_trace()

    #     img = data.cuda()
        
    #     code = test_AE(img)
    #     if i==0:
    #         x = code.view(code.size(0), -1)
    #         img_dis = x.detach().cpu().numpy()
    #     else:
    #         out = code.view(code.size(0), -1)
    #         out = out.detach().cpu().numpy()
    #         img_dis = np.r_[img_dis, out]
            

    #         # img_dis = torch.cat((img_dis, code.view(code.size(0), -1)), 0)
    # # img_dis = img_dis.detach().cpu().numpy()

    
    # f = open(os.path.join(test_path, str(types)+'_AE.txt'), 'w')
    
    # for i  in range(len(img_dis)):
    #     for w in range(len(img_dis[0])):

    #         f.write(str(img_dis[i][w]) + ' ')
    #     f.write('\n')
    # f.close()

    

    # img_dis = np.loadtxt(os.path.join(test_path, str(types)+'_AE.txt'))
    # Cal_img = Data_statistics(img_dis)  
    # # Cal_img.drawing()
    # # print(Cal_img.cov(1))
    # pdb.set_trace()
    # print(Cal_img.cal(256))
    
    

        
        

        



    

