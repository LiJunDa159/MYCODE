import os
import cv2
import numpy as np
from sklearn import decomposition
import tensorflow as tf
from sklearn import preprocessing

corel_path= 'G:/PyCharm/Lunwen3/W_corel'
corel_name=os.listdir(corel_path)

#############################################数据导入#########################################
def corel_read(corel_path):
    imgs = []
    for file in corel_name:
        img = cv2.imread(corel_path + "/" + file)[:, :, ::-1]
        imgs.append(img)
    imgs = np.array(imgs, dtype=float)
    return imgs

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
def pca_feature(datas,ksize,number):
    ksize1=ksize
    ksize2=ksize
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
    Ws=[]
    for i in range(news.shape[0]):
        new2=news[i]
        new2=new2.reshape((ksize1*ksize2,-1))
        pca = decomposition.PCA(n_components=number, copy=True, whiten=False)
        W=pca.fit_transform(new2)
        W=W.reshape((-1,1))
        Ws.append(W)

    Ws=np.array(Ws)
    Ws=Ws.reshape((filter_number,ksize1,ksize2,-1))
    Ws=Ws.transpose((3,1,2,0))

    return Ws

#################################################################################################################
def PCA_feature(datas,number,cut_size1,cut_size2,ksize,pca_number,save_number):
    datas=np.array(datas)
    array_size=int(datas.shape[0]/number)
    ws=[]
    for i in range(number):
        cut=cut_feature(datas[i*array_size:(i+1)*array_size,:,:],cut_size1,cut_size2)
        w=pca_feature(cut,ksize,pca_number)
        w=w.reshape((1,-1))
        ws.append(w)
    ws=np.array(ws)
    ws=ws.reshape((pca_number*number,ksize,ksize,-1))
    ws=ws[0:save_number,:,:,:]

    return ws

###########################################################任意单次卷积####################################################################
def myconv(datas,kernels):
    kernels = kernels.transpose((1, 2, 3, 0))
    feature_imgs = tf.nn.conv2d(datas, kernels, strides=[1, 1, 1, 1], padding='SAME')
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
    W_shape=W.shape
    W=W.reshape((-1, 1))
    W = preprocessing.minmax_scale(W, feature_range=(0, 1), axis=0, copy=True)
    W=W.reshape((W_shape))

    return W

###########################################################测试cifar-10##########################################################
#第一层权重与第一个特征图
corel_data=corel_read(corel_path)
W1=PCA_feature(corel_data, 1, 52, 77, 5, 6, 6)
# W1=z_score(W1)
feature1_imgs=myconv(corel_data, W1)

#第二层权重与第二个特征图
W2=PCA_feature(feature1_imgs,10,86,129,3,4,32)
# W2=z_score(W2)
feature2_imgs=myconv(feature1_imgs,W2)

#第三层权重与第三个特征图
W3=PCA_feature(feature2_imgs,20,86,129,3,4,64)
# W3=z_score(W3)
#
np.save(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W1.npy", W1)
np.save(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W2.npy", W2)
np.save(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W3.npy", W3)

#6,32,64
