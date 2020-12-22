'''
@Descripttion: 
@version: 
@Author: jinlong li
@Date: 2020-07-09 15:19:46
LastEditors: jinlong li
LastEditTime: 2020-09-24 11:56:01
'''

from InfoGAN import Info_D_rectangle, Info_D_square, Info_G_rectangle, Info_G_square, Q_rectangle, Q_square, FrontEnd_rectangle, FrontEnd_square, InfoGAN_Trainer, My_D, My_G, Info_D_square1280, Q_square1280
import threading

from utils import load_data
# from GAN import My_D, My_G, DCGAN_Trainer, Generator, Discriminator



if __name__ == "__main__":
    
    # img_path_rectangle= '/home/admin11/Data_test/Dataset/4.MinImg_training'
    # G_rectangle = Info_G_rectangle()
    # D_rectangle = Info_D_rectangle()
    # FE_rectangle = FrontEnd_rectangle()
    # Q_rectangle = Q_rectangle()

    # Train_rectangle = InfoGAN_Trainer(G_rectangle, FE_rectangle, D_rectangle, Q_rectangle)
    # Train_rectangle.batch_size = 8
    # Train_rectangle.epoch = 100
    # Train_rectangle.training(img_path_rectangle)

    # T1 = threading.Thread(target=Train_rectangle.training, args=(img_path_rectangle,))

   ''' InfoGAN with My_Square '''
    img_path_square= '/home/admin11/Data_test/Dataset/8.MinImg_training'
    G_square = My_G()
    D_square = Info_D_square1280()
    FE_square = My_D()
    Q_square = Q_square1280()

    Train_square = InfoGAN_Trainer(G_square, FE_square, D_square, Q_square)
    Train_square.resize = 0
    Train_square.type = 'test_1'
    # Train_square.dis_c_dim = 4
    Train_square.training(img_path_square)
    

    # T2 = threading.Thread(target=Train_square.training, args=(img_path_square,))

    # T1.start()
    # T2.start()
    # T1.join()
    # T2.join()

    ''' DCGAN with My_rectangle '''

    # rect_G = My_G()
    # rect_D = My_D()
    # # rect_G = Generator()
    # # rect_D = Discriminator()

    # My_DCGAN = DCGAN_Trainer(rect_G, rect_D)
    # My_DCGAN.epoch = 500
    # # My_DCGAN.lr = 0.00001
    # My_DCGAN.image_size = 0
    # My_DCGAN.batch_size = 2
    # # My_DCGAN.dataroot = '/home/admin11/Data_test/test'
    # My_DCGAN.dataroot = '/home/admin11/Data_test/Dataset/7.MinImg_training/1label/'
    # My_DCGAN.img_path = '/home/admin11/1.my_zone/out_imgs/1.DCGAN/My_rect'
    # My_DCGAN.model_path = '/home/admin11/1.my_zone/model/1.DCGAN/My_rect'
    # My_DCGAN.type = 'My_Rect'
    # dataloader = load_data(My_DCGAN.image_size, My_DCGAN.dataroot, My_DCGAN.batch_size)
    # My_DCGAN.training(dataloader) 


