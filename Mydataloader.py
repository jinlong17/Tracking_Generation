'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-09 13:54:06
@LastEditors: jinlong li
@LastEditTime: 2020-07-09 18:12:21
'''

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 加载数据集
def load_data(resize,data,batch_size):
    if int(resize) > 0:
        # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
        data_tf = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    else:
        data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        

    train_dataset =datasets.ImageFolder(data, transform=data_tf)# data 为自己数据集的目录
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader