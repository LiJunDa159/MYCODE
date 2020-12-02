from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
import torch
from RDN import rdn
from LapSRN import *
from DRRN import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/datasets/test')
    parser.add_argument('--dataset', type=str, default='Flickr1024')
    # parser.add_argument('--dataset', type=str, default='Flickr1024_test_x8')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()




def ttest(test_loader, cfg):
    # net=DRRN().to(cfg.device)
    net=rdn().to(cfg.device)
    # net = PASSRnet(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True

    ###导入权重
    pretrained_dict = torch.load('/media/duanting/shuju/PASSRnet-master/log/rdn_epoch_best_val.pth')
    net.load_state_dict(pretrained_dict)
    psnr_list = []
    ssim_list=[]

    with torch.no_grad():
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            video_name = test_loader.dataset.file_list[idx_iter]

            # SR_left = net(LR_left, LR_right, is_training=0) ##PASSRnet要用的

            SR_left = net(LR_left) ##DRRN要用

            SR_left = torch.clamp(SR_left, 0, 1)#归一化　
            psnr_list.append(cal_psnr(HR_left, SR_left))


            HR_left_image=HR_left.permute(2,3,0,1).squeeze()#[H*W*C]
            SR_left_image=SR_left.permute(2,3,0,1).squeeze()
            ssim_list.append(cal_ssim(HR_left_image, SR_left_image))

           
            # save results
            if not os.path.exists('/media/duanting/shuju/PASSRnet-master/result/picture'):
                os.mkdir('result/'+cfg.dataset)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))##[H*W*C] tensor-->PIKLimage
            SR_left_img.save('result/'+'picture'+'/'+video_name+'.png')

            LR_left_img=torch.clamp(LR_left, 0, 1)
            LR_left_img=transforms.ToPILImage()(torch.squeeze(LR_left_img.data.cpu(), 0))
            LR_left_img.save('result/' + 'LRpicture' + '/' + video_name + '.png')

        # print results
        print(cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))
        print(cfg.dataset + ' mean ssim: ', float(np.array(ssim_list).mean()))



def test(test_loader, cfg):
    net=LapSRN().to(cfg.device)
    cudnn.benchmark = True
    pretrained_dict = torch.load('/media/duanting/shuju/PASSRnet-master/log/LapSRN_epoch_best_val.pth')
    net.load_state_dict(pretrained_dict)


    psnr_list2=[]
    psnr_list4=[]
    psnr_list = []

    ssim_list2=[]
    ssim_list4=[]
    ssim_list=[]

    mse_list2 = []
    mse_list4 = []
    mse_list=[]
    with torch.no_grad():
        for idx_iter, (HR_2_left,HR_4_left,HR_left, _, LR_left, _)  in enumerate(test_loader):
            HR_left, LR_left = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device)
            HR_2_left = Variable(HR_2_left).to(cfg.device)
            HR_4_left= Variable(HR_4_left).to(cfg.device)
            video_name = test_loader.dataset.file_list[idx_iter]

            HR_2, HR_4, HR_8 = net(LR_left)
            HR_2 = torch.clamp( HR_2, 0, 1)#归一化　
            HR_4= torch.clamp(HR_4, 0, 1)
            HR_8 = torch.clamp(HR_8, 0, 1)


            psnr_list.append(cal_psnr(HR_left, HR_8))
            psnr_list4.append(cal_psnr(HR_left_4, HR_4))
            psnr_list2.append(cal_psnr(HR_left_2, HR_2))

            HR_left_image=HR_left.permute(2,3,0,1).squeeze()#[H*W*C]
            HR4_left_image = HR_4_left.permute(2, 3, 0, 1).squeeze()
            HR2_left_image = HR_2_left.permute(2, 3, 0, 1).squeeze()

            SR_left_image=HR_8.permute(2,3,0,1).squeeze()
            SR4_left_image=HR_4.permute(2,3,0,1).squeeze()
            SR2_left_image=HR_2.permute(2,3,0,1).squeeze()

            # print(SR_left_image.shape, HR_left_image.shape, '%%%%%')
            ssim_list.append(cal_ssim(HR_left_image, SR_left_image))
            ssim_list4.append(cal_ssim(HR4_left_image, SR4_left_image))
            ssim_list2.append(cal_ssim(HR2_left_image, SR2_left_image))


            # save results
            if not os.path.exists('/media/duanting/shuju/PASSRnet-master/result/picture'):
                os.mkdir('result/'+cfg.dataset)
            SR_left_img = transforms.ToPILImage()(torch.squeeze( HR_8.data.cpu(), 0))##[H*W*C] tensor-->PIKLimage
            SR_left_img.save('result/'+'Lapsrn/'+'picture8'+'/'+video_name+'.png')

            SR_left_img = transforms.ToPILImage()(torch.squeeze(HR_4.data.cpu(), 0))  ##[H*W*C] tensor-->PIKLimage
            SR_left_img.save('result/' +'Lapsrn/'+ 'picture4' + '/' + video_name + '.png')

            SR_left_img = transforms.ToPILImage()(torch.squeeze(HR_2.data.cpu(), 0))  ##[H*W*C] tensor-->PIKLimage
            SR_left_img.save('result/' + 'Lapsrn/' + 'picture2' + '/' + video_name + '.png')

            LR_left_img=torch.clamp(LR_left, 0, 1)
            LR_left_img=transforms.ToPILImage()(torch.squeeze(LR_left_img.data.cpu(), 0))
            LR_left_img.save('result/' + 'Lapsrn/'+'LRpicture' + '/' + video_name + '.png')

        # print results
        print(cfg.dataset + ' mean psnr8: ', float(np.array(psnr_list).mean()))
        print(cfg.dataset + ' mean psnr4: ', float(np.array(psnr_list4).mean()))
        print(cfg.dataset + ' mean psnr2: ', float(np.array(psnr_list2).mean()))

        print(cfg.dataset + ' mean ssim8: ', float(np.array(ssim_list).mean()))
        print(cfg.dataset + ' mean ssim4: ', float(np.array(ssim_list4).mean()))
        print(cfg.dataset + ' mean ssim2: ', float(np.array(ssim_list2).mean()))



def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=4)
    # test_set = TestSetLoader(dataset_dir=cfg.testset_dir , scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    result = ttest(test_loader, cfg)
    # result = test(test_loader, cfg)  ###用ttest时注释
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    from torch import nn
