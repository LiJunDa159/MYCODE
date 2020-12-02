import cv2
import numpy as np
def yz(img):
    img_array = np.array(img).astype(np.float32)#转化成数组
    I=img_array
    zmax=np.max(I)
    zmin=np.min(I)
    tk=(zmax+zmin)/2#设置初始阈值
    #根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度  zo和zb
    b=1
    m,n=I.shape
    while b==0:
        ifg=0
        ibg=0
        fnum=0
        bnum=0
        for i in range(1,m):
             for j in range(1,n):
                tmp=I(i,j)
                if tmp>=tk:
                    ifg=ifg+1
                    fnum=fnum+int(tmp)  #前景像素的个数以及像素值的总和
                else:
                    ibg=ibg+1
                    bnum=bnum+int(tmp)#背景像素的个数以及像素值的总和
        #计算前景和背景的平均值
        zo=int(fnum/ifg)
        zb=int(bnum/ibg)
        if tk==int((zo+zb)/2):
            b=0
        else:
            tk=int((zo+zb)/2)

    for i in range(1, m):
        for j in range(1, n):
            if I[i, j] >= tk:
                I[i, j] = int(I[i, j])
            else:
                I[i, j] = int(I[i, j]*0.8)
    img = I
    return img


def yz3(img):
    img_b,img_g,img_r=cv2.split(img)   #分离三通道
    img_b=yz(img_b)
    img_g=yz(img_g)
    img_r=yz(img_r)
    img=cv2.merge([img_b,img_g,img_r])
    return img