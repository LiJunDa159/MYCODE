import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle

##############################################################数据载入############################################################################################
def load(data_path, image_size1, image_size2,classes):
    images = []     #图像
    labels = []     #标签
    img_names = []  #图像名
    cls = []        #类别

    # print('Going to read images')
    for fields in classes:   
        index = classes.index(fields)
        # print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(data_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size2, image_size1),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            # image = np.multiply(image, 1.0 / 255.0)#归一化
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)#图像名字
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)#图像与标签对应
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

##################################################################################数据抽取（next_batch）###################################################################
class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]  #属性定义
    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
    assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

################################################################训练数据集构造（训练集与验证集）###########################################################################
def read_train_sets(train_path, image_size1, image_size2, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load(train_path, image_size1, image_size2, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

  if isinstance(validation_size, float):#验证集
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)#指定一个类将信息组合
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets

###########################################################测试数据集#####################################################################
def read_test_sets(test_path, image_size1, image_size2,classes):
  class DataSets(object):
    pass

  data_sets = DataSets()

  test_images, test_labels, test_img_names, test_cls = load(test_path, image_size1, image_size2, classes)

  data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)

  return data_sets

############################################################数据类型查看#########################################################################
def show_test_data(data_sets):
    print('{test展示结果')
    print('data_sets_type:\t{}'.format(type(data_sets)))
    print('data_type:\t{}'.format(type(data_sets.test.images)))
    print('data_shape:\t{}'.format(data_sets.test.images.shape))
    print('data[0]_shape:\t{}'.format(data_sets.test.images[0].shape))
    print('data[0,:,:,0]_shape:\t{}'.format(data_sets.test.images[0,:,:,0].shape))
    print('data[0,:,:,0]_type:\t{}'.format(type(data_sets.test.images[0,:,:,0])))
    print('data[0][0][0][0]\t{}'.format(data_sets.test.images[0][0][0][0]))
    print('label_type:\t{}'.format(type(data_sets.test.labels)))
    print('label_shape:\t{}'.format(data_sets.test.labels.shape))
    print('label[0]:\t{}'.format(data_sets.test.labels[0]))
    return

def show_train_data(data_sets):
    print('train展示结果')
    print('data_sets_type:\t{}'.format(type(data_sets)))
    print('data_type:\t{}'.format(type(data_sets.train.images)))
    print('data_shape:\t{}'.format(data_sets.train.images.shape))
    print('data[0]_shape:\t{}'.format(data_sets.train.images[0].shape))
    print('data[0,:,:,0]_shape:\t{}'.format(data_sets.train.images[0,:,:,0].shape))
    print('data[0,:,:,0]_type:\t{}'.format(type(data_sets.train.images[0,:,:,0])))
    print('data[0][0][0][0]\t{}'.format(data_sets.train.images[0][0][0][0]))
    print('label_type:\t{}'.format(type(data_sets.train.labels)))
    print('label_shape:\t{}'.format(data_sets.train.labels.shape))
    print('label[0]:\t{}'.format(data_sets.train.labels[0]))
    print('data_valid_shape:\t{}'.format(data_sets.valid.images.shape))
    return

##############################################################图像查看##############################################################
# classes = ['n0','n1', 'n2','n3','n4','n5','n6','n7','n8','n9']
# num_classes = len(classes)
# cifar_img_size1 =32
# cifar_img_size2=32
# cifar_path='W_cifar'
# validation_size=0.2
#
# train_cifar_data=read_train_sets(cifar_path, cifar_img_size1, cifar_img_size2, classes,validation_size )
# test_cifar_data=read_test_sets(cifar_path, cifar_img_size1, cifar_img_size2, classes )
# show_train_data(train_cifar_data)
# show_test_data(test_cifar_data)