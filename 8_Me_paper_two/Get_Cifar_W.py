import os
import cv2
import numpy as np
from sklearn import decomposition
import tensorflow as tf
import time


cifar_path='G:/PyCharm/Lunwen3/W_cifar'
cifar_name=os.listdir(cifar_path)
ksize1=5
ksize2=5

#3*3,11
#5*5,7

#############################################数据导入#########################################
def cifar_read(cifar_path):
    imgs = []
    for file in cifar_name:
        img = cv2.imread(cifar_path + "/" + file)[:, :, ::-1]
        imgs.append(img)
    imgs = np.array(imgs, dtype=float)
    return imgs

#######################################平滑滤波######################################################
def ave_img(imgs):
    ave_imgs=[]
    for i in range(imgs.shape[0]):
        ave_img= cv2.blur(imgs[i],(5,5))
        ave_imgs.append(ave_img)

    return ave_imgs

##############################################边缘检测##########################################################
def sobel_img(imgs):
    sobel_imgs=[]
    for i in range(imgs.shape[0]):
        sobel_img=cv2.Sobel(imgs[i],cv2.CV_64F,1,1,ksize=5)
        sobel_imgs.append(sobel_img)
    return sobel_imgs

#################################################图像切割########################################################
def divide_method2(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)
    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列）

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image

def cut_imgs(ave_data,sobel_data,data):
    ave_data=np.array(ave_data)
    sobel_data=np.array(sobel_data)
    data=np.array(data)
    print(data.shape)
    cut_ave_datas=[]
    cut_sobel_datas=[]
    cut_datas=[]
    for i in range(len(data)):
        cut_ave_data= divide_method2(ave_data[i],7,7)
        cut_ave_datas.append(cut_ave_data)
        cut_sobel_data = divide_method2(sobel_data[i],7,7)
        cut_sobel_datas.append(cut_sobel_data)
        cut_data = divide_method2(data[i],7,7)
        cut_datas.append(cut_data)

    return cut_ave_datas, cut_sobel_datas,cut_datas

#####################################################PCA################################################################
def PCA(datas):
    ksize1 = 5
    ksize2 = 5
    datas=np.array(datas)
    new1s=[]
    for i in range(datas.shape[0]):
        for j in range(datas[i].shape[1]):
            for k in range(datas[i].shape[2]):
                new1=datas[i,j,k,:,:,:]
                new1s.append(new1)        #去前三维，只保留卷积核大小和通道数

    new1_Bs=[]
    new1_Gs=[]
    new1_Rs=[]

    for i in range(len(new1s)):
        new1_B,new1_G,new1_R = cv2.split(new1s[i])
        new1_Bs.append(new1_B)
        new1_Gs.append(new1_G)
        new1_Rs.append(new1_R)         #只保留了1*9的卷积核

    new1_Bs=np.array(new1_Bs).reshape((ksize1*ksize2,-1))
    new1_Gs=np.array(new1_Gs).reshape((ksize1*ksize2,-1))
    new1_Rs=np.array(new1_Rs).reshape((ksize1*ksize2,-1))

    pca = decomposition.PCA(n_components=0.65, copy=True, whiten=True)
    W1_B = pca.fit_transform(new1_Bs)
    W1_B=np.transpose(W1_B)
    W1_B=W1_B.reshape(-1,ksize1,ksize2)
    # print('W1_B_type:\t{}'.format(W1_B.shape))
    W1_G = pca.fit_transform(new1_Gs)
    W1_G = np.transpose(W1_G)
    W1_G=W1_G.reshape(-1,ksize1,ksize2)
    # print('W1_G_type:\t{}'.format(W1_G.shape))
    W1_R = pca.fit_transform(new1_Rs)
    W1_R = np.transpose(W1_R)
    W1_R=W1_R.reshape(-1,ksize1,ksize2)
    # print('W1_R_type:\t{}'.format(W1_R.shape))
    l=min([W1_B.shape[0],W1_G.shape[0],W1_R.shape[0]])
    W1_B = W1_B[0:l, :, :]
    W1_G = W1_G[0:l, :, :]
    W1_R = W1_R[0:l, :, :]
    W1=np.array([W1_B,W1_G,W1_R])
    W1=W1.transpose((1,2,3,0))

    return W1

###########################################################任意单次卷积####################################################################
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def myconv(datas,kernels):
    kernels = kernels.transpose((1, 2, 3, 0))
    feature_imgs=tf.nn.conv2d(datas, kernels, strides=[1, 1, 1, 1], padding='SAME')
    feature_imgs = tf.nn.relu(feature_imgs)
    feature_imgs = tf.nn.max_pool(value=feature_imgs,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feature_imgs = feature_imgs.eval(session=sess)

    return feature_imgs
#########################################################归一化########################################################
def z_score(W):
    W=np.array(W)
    # W_shape = W.shape
    W=W.transpose((1,2,3,0))
    w=tf.truncated_normal(W.shape, mean=0, stddev=0.05)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    w= w.eval(session=sess)
    W=w
    W=W.transpose((3,0,1,2))
    # w=w.reshape(-1)
    # W=W.reshape((-1, 1))
    # for i in range(W.shape[0]):
    #    if abs(W[i])>= 0:
    #        W[i]=w[i]
    #    else:
    #        W[i]=W[i]
    # W=W.reshape((W_shape))

    return W
###############################################################cut_data#############################################################
def divide_method3(data, m, n):  # 分割成m行n列

    h, w = data.shape[0], data.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (n - 1) + 0.5)
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)
    data=data[0:h,0:w]
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)
    New_data = np.zeros([m - 1, n - 1, grid_h, grid_w, data.shape[2]],np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列）

    for i in range(m - 1):
        for j in range(n - 1):
            New_data[i, j, ...] = data[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]

    return New_data


def cut_feature(datas,cut_size1,cut_size2):
    cut_datas=[]
    for i in range(len(datas)):
        cut_data=divide_method3(datas[i],cut_size1,cut_size2)
        cut_datas.append(cut_data)

    return cut_datas

#############################################################PCA_feature##############################################################
def pca_feature(datas,number):
    ksize1 = 3
    ksize2 = 3
    datas = np.array(datas)
    filter_number=datas.shape[5]
    news = []
    for i in range(datas.shape[0]):
        for j in range(datas.shape[1]):
            for k in range(datas.shape[2]):
                new1 = datas[i, j, k, :, :, :]
                news.append(new1)  # 去前三维，只保留卷积核大小和通道数,现在news是四维

    news=np.array(news)
    news=news.transpose((3,1,2,0))
    # print('news:\t{}'.format(news.shape))
    Ws=[]
    for i in range(news.shape[0]):
        new2=news[i]
        new2=new2.reshape((ksize1*ksize2,-1))
        pca = decomposition.PCA(n_components=number, copy=True, whiten=False)
        W=pca.fit_transform(new2)
        W=W.reshape((1,-1))
        Ws.append(W)

    Ws = np.array(Ws)
    Ws=Ws.reshape((filter_number,ksize1,ksize2,-1))
    Ws=Ws.transpose((3,1,2,0))

    return Ws

def PCA_feature(datas,number,cutsize1,cutsize2,pca_number,save_number):
    datas=np.array(datas)
    array_size=int(datas.shape[0]/number)
    ws=[]
    for i in range(number):
        cut=cut_feature(datas[i*array_size:(i+1)*array_size,:,:],cutsize1,cutsize2)
        w=pca_feature(cut,pca_number)
        w=w.reshape((1,-1))
        ws.append(w)
    ws=np.array(ws)
    ws=ws.reshape((pca_number*number,3,3,-1))
    ws=ws[0:save_number,:,:,:]

    return ws

#################################################################cifar测验############################################################
start = time.perf_counter()    #起始时间
#第一层权重与第一个特征图
cifar_data=cifar_read(cifar_path)
ave_cifar_data=ave_img(cifar_data)
sobel_cifar_data=sobel_img(cifar_data)
cut_ave_cifar_data,cut_sobel_cifar_data,cut_cifar_data=cut_imgs(ave_cifar_data,sobel_cifar_data,cifar_data)

W1_ave=PCA(cut_ave_cifar_data)
W1_sobel=PCA(cut_sobel_cifar_data)
W1_data=PCA(cut_cifar_data)

W1=np.vstack((W1_ave,W1_sobel,W1_data))
W1=W1[0:32,:,:,:]
# W1=z_score1(W1)
feature1_imgs=myconv(cifar_data,W1)

#第二层权重与第二个特征图
W2=PCA_feature(feature1_imgs,10,11,11,7,64)
# W2=z_score1(W2)
feature2_imgs=myconv(feature1_imgs,W2)

#第三层权重与第二个特征图
W3=PCA_feature(feature2_imgs,20,11,11,7,128)
# W3=z_score1(W3)

np.save(r"G:\PyCharm\Lunwen3\Ws_cifar\W1.npy", W1)
np.save(r"G:\PyCharm\Lunwen3\Ws_cifar\W2.npy", W2)
np.save(r"G:\PyCharm\Lunwen3\Ws_cifar\W3.npy", W3)

end = time.perf_counter()   #结束时间
print('%.5f'%(end - start))   #输出运行时间，精确到小数点后5位

#32,64,128

