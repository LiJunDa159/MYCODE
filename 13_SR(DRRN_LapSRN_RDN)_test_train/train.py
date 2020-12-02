from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
from datetime import datetime
from RDN import rdn
from LapSRN import *
from DRRN import *
from PASSRnet import *
###用的时候可以把不用的先注释

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/train')
    # parser.add_argument('--trainset_dir', type=str, default='data/datasets/train/Flickr1024_patches')
    parser.add_argument('--valset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/val/Flickr1024_val_x8')

    return parser.parse_args()

#####LapSRN多监督训练函数××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

def train(train_loader,val_loader,cfg):
    net=LapSRN().to(cfg.device)
    cudnn.benchmark = True
    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    # criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    prec_time = datetime.now()
    loss_epoch = []
    loss_list = []
    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left_2,HR_left_4,HR_left, _, LR_left, LR_right) in tqdm( enumerate(train_loader), total=len(train_loader)):
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device).cuda(), Variable(LR_left).to(cfg.device).cuda(), Variable(LR_right).to(cfg.device).cuda()
            # SR_left=net(LR_left)
            HR_left_2=Variable(HR_left_2).to(cfg.device).cuda()
            HR_left_4=Variable(HR_left_4).to(cfg.device).cuda()

            HR_2, HR_4, HR_8=net(LR_left)
            ### loss_SR
            loss2=criterion_mse(HR_2,HR_left_2)
            loss4=criterion_mse(HR_4,HR_left_4)
            loss8=criterion_mse(HR_8,HR_left)
            loss_SR=loss2+loss4+loss8

            optimizer.zero_grad()
            loss_SR.backward()
            optimizer.step()
            loss_epoch.append(loss_SR.data.cpu())
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print('Epoch----%5d, train_loss---%f ' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            loss_epoch = []


# ############val##############验证时也可以不计算loss####################
        psnr_epoch2=[]
        psnr_epoch4=[]
        psnr_epoch = []
        lossepoch =[]
        net = net.eval()
        for idx, (HR_2_left,HR_4_left,HRleft, _, LRleft, _) in tqdm(enumerate(val_loader),total=len(val_loader)):
            HRleft, LRleft  = Variable(HRleft).to(cfg.device), Variable(LRleft).to(cfg.device)
       
            HR_left_2=Variable(HR_2_left).to(cfg.device)
            HR_left_4 = Variable(HR_4_left).to(cfg.device)
            ### loss_SR
            HR_2, HR_4, HR_8 = net(LRleft)

            # print(HR_2.shape,HR_4.shape,HR_8.shape,HR_2_left.shape,HR_4_left.shape,HRleft.shape,'$$$$$$$$$$$$$$')

            loss2 = criterion_mse(HR_2,  HR_left_2)
            loss4 = criterion_mse(HR_4, HR_left_4)
            loss8= criterion_mse(HR_8, HRleft)
            lossSR = loss2 + loss4 + loss8
            optimizer.zero_grad()
            lossSR.backward()
            optimizer.step()
            lossepoch.append(lossSR.data.cpu())

            psnr2=cal_psnr(HR_left_2[:, :, :, 64:].data.cpu(), HR_2[:, :, :, 64:].data.cpu())
            psnr_epoch2.append(psnr2)

            psnr4 = cal_psnr(HR_left_4.data.cpu(), HR_4.data.cpu())
            psnr_epoch4.append(psnr4)

            psnr = cal_psnr(HRleft[:, :, :, 64:].data.cpu(), HR_8[:, :, :, 64:].data.cpu())
            psnr_epoch.append(psnr)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)

        if idx_epoch % 1 == 0:
            best = [0]
            if (max(best) <= float(np.array(psnr_epoch2).mean())):
                best.append(float(np.array(psnr_epoch2).mean()))
                torch.save(net.state_dict(), '/media/duanting/shuju/PASSRnet-master/log/LapSRN_epoch_best_val.pth')
            print('Epoch----%5d, val_loss---%f,PSNR2---%f,PSNR4---%f,PSNR8---%f' % (idx_epoch + 1, float(np.array(lossepoch).mean()),
                                                                                     float(np.array(psnr_epoch2).mean()),
                                                                                    float(np.array(psnr_epoch4).mean()),
                                                                                    float(np.array(psnr_epoch).mean()),
                                                                                    ))


            for i in best:
                sum=0
                sum=sum+i
            print('best_psnr',sum/(len(best)-1))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print('time:',time_str)


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    tra = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    val_set = ValSetLoader(dataset_dir=cfg.valset_dir,scale_factor=8)
    val = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    train(tra,val,cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

##########################################################################


#####DRRN单监督训练×××××××××××××××××××××××××××××××××××××××××××××××××××××××

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/train/Flickr1024_patches')
    parser.add_argument('--valset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/val')

    return parser.parse_args()


def train(train_loader,val_loader,cfg):
    net=DRRN().to(cfg.device)
    # net.apply(weights_init_xavier)
    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    # criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    prec_time = datetime.now()
    loss_epoch = []
    loss_list = []
    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, _, LR_left, LR_right) in tqdm( enumerate(train_loader), total=len(train_loader)):
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device).cuda(), Variable(LR_left).to(cfg.device).cuda(), Variable(LR_right).to(cfg.device).cuda()
            SR_left=net(LR_left)
            loss_SR=criterion_mse(SR_left,HR_left)
            optimizer.zero_grad()
            loss_SR.backward()
            optimizer.step()
            loss_epoch.append(loss_SR.data.cpu())


        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print('Epoch----%5d, train_loss---%f ' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))

            loss_epoch = []


# ############val##############
        psnr_epoch = []
        lossepoch =[]
        net = net.eval()
        for idx, (HR_left, _, LR_left, LR_right) in tqdm(enumerate(val_loader),total=len(val_loader)):
            HRleft, LRleft  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device)
            SRleft=net(LRleft)

            ### loss_SR
            lossSR = criterion_mse(SRleft, HRleft)

            optimizer.zero_grad()
            lossSR.backward()
            optimizer.step()
            lossepoch.append(lossSR.data.cpu())

            psnr = cal_psnr(HRleft[:, :, :, 64:].data.cpu(), SRleft[:, :, :, 64:].data.cpu())
            psnr_epoch.append(psnr)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)

        if idx_epoch % 1 == 0:
            best = [0]
            if (max(best) <= float(np.array(psnr_epoch).mean())):
                best.append(float(np.array(psnr_epoch).mean()))
                torch.save(net.state_dict(), '/media/duanting/shuju/PASSRnet-master/log/DRRN93_epoch_best_val.pth')
            print('Epoch----%5d, val_loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(lossepoch).mean()),
                                                                   float(np.array(psnr_epoch).mean())))

            for i in best:
                sum=0
                sum=sum+i
            print('best_psnr',sum/(len(best)-1))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print('time:',time_str)



def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    tra = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    val_set = ValSetLoader(dataset_dir=cfg.valset_dir,scale_factor=4)
    val = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    # print(val)
    train(tra,val,cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
###############################################RDN#####################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/train/Flickr1024_patches')
    parser.add_argument('--valset_dir', type=str, default='/media/duanting/shuju/PASSRnet-master/data/datasets/val')


    return parser.parse_args()

class L1_Charboonier_loss(nn.Module):
    def __init__(self):
        super(L1_Charboonier_loss,self).__init__()
        self.eps = 1e-6
    def forward(self,x,y):
        diff = torch.add(x,-y)
        error = torch.sqrt(diff*diff+self.eps)
        loss = torch.sum(error)
        return loss

def train(train_loader,val_loader,cfg):
    net=rdn().to(cfg.device)

    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device) 
    # criterion_L1 = L1Loss()
    criterion=L1_Charboonier_loss().to(cfg.device)


    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    prec_time = datetime.now()
    loss_epoch = []
    loss_list = []
    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, _, LR_left, LR_right) in tqdm( enumerate(train_loader), total=len(train_loader)):
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device).cuda(), Variable(LR_left).to(cfg.device).cuda(), Variable(LR_right).to(cfg.device).cuda()
            SR_left=net(LR_left)
            loss_SR=criterion_mse (SR_left,HR_left)
            optimizer.zero_grad()
            loss_SR.backward()
            optimizer.step()
            loss_epoch.append(loss_SR.data.cpu())


        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print('Epoch----%5d, train_loss---%f ' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))

            loss_epoch = []


# ############val##############
        psnr_epoch = []
        lossepoch =[]
        net = net.eval()
        for idx, (HR_left, _, LR_left, LR_right) in tqdm(enumerate(val_loader),total=len(val_loader)):
            HRleft, LRleft  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device)
            SRleft=net(LRleft)

            ### loss_SR
            lossSR = criterion_mse (SRleft, HRleft)
            optimizer.zero_grad()
            lossSR.backward()
            optimizer.step()
            lossepoch.append(lossSR.data.cpu())

            psnr = cal_psnr(HRleft[:, :, :, 64:].data.cpu(), SRleft[:, :, :, 64:].data.cpu())
            psnr_epoch.append(psnr)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)

        if idx_epoch % 1 == 0:
            best = [0]
            if (max(best) <= float(np.array(psnr_epoch).mean())):
                best.append(float(np.array(psnr_epoch).mean()))
                torch.save(net.state_dict(), '/media/duanting/shuju/PASSRnet-master/log/rdn_epoch_best_val.pth')
            print('Epoch----%5d, val_loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(lossepoch).mean()),
                                                                   float(np.array(psnr_epoch).mean())))

        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print('time:',time_str)



def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    tra = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    val_set = ValSetLoader(dataset_dir=cfg.valset_dir,scale_factor=4)
    val = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    train(tra,val,cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)




###############################################PASSRnet#################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='data/datasets/train/Flickr1024_patches')
    return parser.parse_args()


def train(train_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    # net.apply(weights_init_xavier)
    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(train_loader):

            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = net(LR_left, LR_right, is_training=1)

      
            loss_SR = criterion_mse(SR_left, HR_left)
          
            ### loss_smoothness
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ### loss_cycle
            Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).to(cfg.device)
            loss_cycle = criterion_L1(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3), Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                         criterion_L1(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

            ### loss_photometric
            LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b*h,w,w), LR_right.permute(0,2,3,1).contiguous().view(b*h, w, c))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = criterion_L1(LR_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
                          criterion_L1(LR_right * V_right_to_left, LR_left_warped * V_right_to_left)

            ### losses
            loss = loss_SR + 0.005 * (loss_photo + loss_smooth + loss_cycle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss_SR.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))


            torch.save(net.state_dict(),  '/home/duanting/Documents/super resoluti0n/PASSRnet-master/log'+'/PASSnet_epoch' + str(idx_epoch + 1) + '.pth')
            psnr_epoch = []
            loss_epoch = []

def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)




