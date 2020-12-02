import matplotlib.pyplot as plt
import cv2
import numpy as np
#####################################################################################
def ave_img(imgs):
    ave_imgs= cv2.blur(imgs,(25,25))
    return ave_imgs
######################################################################################
def sobel_img(imgs):
    imgs=cv2.cvtColor(imgs,cv2.COLOR_RGB2GRAY)
    sobel_x=cv2.Sobel(imgs,cv2.CV_64F,1,0,ksize=5)
    sobel_x=cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(imgs, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    np.set_printoptions(threshold=np.inf)
    sobel_imgs =cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    return sobel_imgs
######################################################################################
def divide_method(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                            np.uint8)

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image
######################################################################################
def divide_method2(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w],
                            np.uint8)

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1]]
    return divide_image
###########################################################################################
def display_blocks(divide_image):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            plt.axis('off')
            # plt.title('block:'+str(i*n+j+1))
    plt.show()
#########################################################################################
def display_blocks2(divide_image):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j],'gray')
            plt.axis('off')
            # plt.title('block:'+str(i*n+j+1))
    plt.show()
##############################################################################################
img = cv2.imread('132.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_ave=ave_img(img)
img_sobel=sobel_img(img)
fig1 = plt.figure('原始图像')
plt.imshow(img)
plt.axis('off')
plt.title('Original image',fontdict={'weight':'normal','size': 23})
plt.show()
fig2 = plt.figure('模糊图像')
plt.imshow(img_ave)
plt.axis('off')
plt.title('Blurred image',fontdict={'weight':'normal','size': 23})
plt.show()
fig3 = plt.figure('边缘图像')
plt.imshow(img_sobel,'gray')
plt.axis('off')
plt.title('Edge image',fontdict={'weight':'normal','size': 23})
plt.show()
cut_img=divide_method(img,4,5)
cut_ave=divide_method(img_ave,4,5)
cut_sobel=divide_method2(img_sobel,4,5)
display_blocks(cut_img)
display_blocks(cut_ave)
display_blocks2(cut_sobel)




