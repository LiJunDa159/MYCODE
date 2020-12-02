from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
import skimage
from skimage import measure
import skimage.transform
from torch.nn import init
from PIL import Image, ImageFilter
import cv2
from torchvision.transforms import CenterCrop
def bicubicc(lr_image, upscale):
    w, h = lr_image.size
    w = upscale * w
    h = upscale * h

    return lr_image.resize((w, h), Image.CUBIC)

def suo(img,scale):
    w,h=img.size
    w=int(w/scale)
    h=int(h/scale)
    return img.resize((w, h), Image.CUBIC)

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        # img_lr_left1=bicubicc(img_lr_left,4) ###双三次插值放大倍数
    


	#多监督要用到的
        # img_lr_left=suo(img_lr_left,2) ###双三次插值缩小倍数
        # img_hr_left2=suo(img_hr_left,4)
        # img_hr_left4=suo(img_hr_left,2)
        # img_hr_left2=np.array(img_hr_left2,dtype=np.float32)
        # img_hr_left4=np.array(img_hr_left4,dtype=np.float32)


        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)


        # img_hr_left2,img_hr_left4,img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation1(img_hr_left2,img_hr_left4,img_hr_left, img_hr_right, img_lr_left, img_lr_right) #多监督要用到的
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)


        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
        # return toTensor(img_hr_left2),toTensor(img_hr_left4),toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)#多监督要用到的

    def __len__(self):
        return len(self.file_list)





class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.i = 0
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor

        # self.file_list = os.listdir(dataset_dir + '/lr_x8')
        self.file_list = os.listdir(dataset_dir + '/lr_x4')
    def __getitem__(self, index):
        print(self.file_list,'***********')
        hr_image_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png')


        # lr_image_left=bicubicc(lr_image_left,4)
     
        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')

        # hr_image_left2 = suo(hr_image_left, 4)
        # hr_image_left4 = suo(hr_image_left, 2)
        #
        #
        # hr_image_left2 = ToTensor()(hr_image_left2)
        # hr_image_left4 = ToTensor()(hr_image_left4)


        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)


        return  hr_image_left, hr_image_right, lr_image_left, lr_image_right  #[C*H*W]
        # return  hr_image_left2,hr_image_left4,hr_image_left, hr_image_right, lr_image_left, lr_image_right  #[C*H*W]

    def __len__(self):
        return len(self.file_list)


class ValSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(ValSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
     
        hr_image_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png')

        # lr_image_left=bicubicc(lr_image_left,4)


        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')


        # hr_image_left2=suo(hr_image_left,4)
        # hr_image_left4=suo(hr_image_left,2)
        #
        # hr_image_left2=ToTensor()(hr_image_left2)
        # hr_image_left4=ToTensor()( hr_image_left4)


        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)

       
        # return  hr_image_left2,hr_image_left4,hr_image_left, hr_image_right, lr_image_left, lr_image_right  #[C*H*W]
        return  hr_image_left, hr_image_right, lr_image_left, lr_image_right  #[C*H*W]
    def __len__(self):
        return len(self.file_list)




def augumentation1(hr_image_left2,hr_image_left4,hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_left2=hr_image_left2[:, ::-1, :]
            hr_image_left4=hr_image_left4[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_left2 = hr_image_left2[::-1, :, :]
            hr_image_left4 = hr_image_left4[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]
        """"no rotation
        if random.random()<0.5:
            lr_image_left = lr_image_left.transpose(1, 0, 2)
            lr_image_right = lr_image_right.transpose(1, 0, 2)
            hr_image_left = hr_image_left.transpose(1, 0, 2)
            hr_image_right = hr_image_right.transpose(1, 0, 2)
        """
        return np.ascontiguousarray(hr_image_left2),np.ascontiguousarray(hr_image_left4),np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)


def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
    if random.random() < 0.5:  # flip horizonly
        lr_image_left = lr_image_left[:, ::-1, :]
        lr_image_right = lr_image_right[:, ::-1, :]
        hr_image_left = hr_image_left[:, ::-1, :]
        hr_image_right = hr_image_right[:, ::-1, :]
    if random.random() < 0.5:  # flip vertically
        lr_image_left = lr_image_left[::-1, :, :]
        lr_image_right = lr_image_right[::-1, :, :]
        hr_image_left = hr_image_left[::-1, :, :]
        hr_image_right = hr_image_right[::-1, :, :]
    """"no rotation
    if random.random()<0.5:
        lr_image_left = lr_image_left.transpose(1, 0, 2)
        lr_image_right = lr_image_right.transpose(1, 0, 2)
        hr_image_left = hr_image_left.transpose(1, 0, 2)
        hr_image_right = hr_image_right.transpose(1, 0, 2)
    """
    return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
           np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def cal_psnr(img1, img2):
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    # print(img1_np.shape,'&&&&&&&&&')
    return measure.compare_psnr(img1_np, img2_np)

def cal_ssim(img1, img2):
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    # ssim = skimage.measure.compare_ssim(img1_np, img2_np,data_range=255,multichannel=True)
    ssim = skimage.measure.compare_ssim(img1_np, img2_np,multichannel=True)
    return ssim




