import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
# conda install --channel https://conda.anaconda.org/menpo opencv3
#Adding Seed so that random initialization is consistent
from numpy.random import seed   #随机种子，对数据切分，保证测试与训练集不随调试变化而变化
seed(10)  #10貌似像名字，设成什么都行
from tensorflow import set_random_seed
set_random_seed(20)


batch_size = 80

#Prepare input data
classes = ['n0','n1', 'n2','n3','n4','n5','n6','n7','n8','n9']
# classes = ['dogs','cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2 #如何切分
img_size = 64
num_channels = 3
train_path='training_data'
logs_path='G:\PyCharm\小论文2\logs'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)#？？？



##Network graph params
filter_size_conv1 = 7
num_filters_conv1 = 64

filter_size_conv2 = 5
num_filters_conv2 = 128

filter_size_conv3 = 3
num_filters_conv3 = 256

filter_size_conv4 = 3
num_filters_conv4 = 256

filter_size_conv5 = 3
num_filters_conv5 = 256

fc_layer_size = 64*32

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


#####################################################################################################################################
def create_convolutional_layer_1(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 2, 2, 1],
                     padding='SAME')

    layer += biases
    
    layer = tf.nn.relu(layer)
    
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    #layer = tf.nn.relu(layer)

    return layer
####################################################################################################################
def create_convolutional_layer_2(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    # layer = tf.nn.relu(layer)

    return layer
#########################################################################################################################
def create_convolutional_layer_34(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    return layer
#########################################################################################################################
def create_convolutional_layer_5(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    return layer
######################################################################################################################

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()#乘积

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer

################################################################################################################################
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    
    layer=tf.nn.dropout(layer,keep_prob=0.98)
    
    if use_relu:
        layer = tf.nn.relu(layer)
        

    return layer

####################################################################################################################################

layer_conv1 = create_convolutional_layer_1(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)#这是一个计算过程

tf.summary.histogram('layer_conv1', layer_conv1)

layer_conv2 = create_convolutional_layer_2(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

tf.summary.histogram('layer_conv2', layer_conv2)

layer_conv3= create_convolutional_layer_34(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

tf.summary.histogram('layer_conv3', layer_conv3)

layer_conv4= create_convolutional_layer_34(input=layer_conv3,
               num_input_channels=num_filters_conv3,
               conv_filter_size=filter_size_conv4,
               num_filters=num_filters_conv4)

tf.summary.histogram('layer_conv4', layer_conv4)

layer_conv5= create_convolutional_layer_5(input=layer_conv4,
               num_input_channels=num_filters_conv4,
               conv_filter_size=filter_size_conv5,
               num_filters=num_filters_conv5)

tf.summary.histogram('layer_conv5', layer_conv5)

layer_flat = create_flatten_layer(layer_conv5)#拉长

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

tf.summary.histogram('layer_fc1', layer_fc1)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=fc_layer_size,
                     use_relu=True)

tf.summary.histogram('layer_fc2', layer_fc2)

layer_fc3 = create_fc_layer(input=layer_fc2,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

tf.summary.histogram('layer_fc3', layer_fc3)

with tf.name_scope('Y_pred'):
    y_pred = tf.nn.softmax(layer_fc3,name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)
with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                    labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())
#记录张量的数据
tf.summary.image('preprocess', tf.reshape(x, [-1, 32, 32, 3]), 10)
tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()   #定义记录运算
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())   #创建写对象


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1,i, acc, val_acc, val_loss))

total_iterations = 0#指定初始迭代次数

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations# 保证更新全局变量.
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples/batch_size) == 0:
            _, loss_, summary = session.run([optimizer, cost, merged_summary_op],
                                            feed_dict=feed_dict_val)  # 执行记录运算
            val_loss = session.run(cost, feed_dict=feed_dict_val)

            epoch = int(i / int(data.train.num_examples/batch_size))  #  data.train.num_examples=800
            summary_writer.add_summary(summary, i)  # 将日志写入文件
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,i)
            saver.save(session, './dogs-cats-model/save_model.pd',global_step=i) #保存模型


    total_iterations += num_iteration

train(num_iteration=51)

# #cd G:\PyCharm\Lunwen2
# G:
# activate tensorflow
# tensorboard --logdir=logs --host=127.0.0.1
# http://127.0.0.1:6006