import dataset
import tensorflow as tf
import numpy as np
from numpy.random import seed  # 随机种子，对数据切分，保证测试与训练集不随调试变化而变化

seed(10)  # 10貌似像名字，设成什么都行
from tensorflow import set_random_seed

set_random_seed(20)

batch_size = 64

# Prepare input data
classes = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
num_classes = len(classes)

validation_size = 0.2  # 如何切分
img_size1 = 64
img_size2 = 96
num_channels = 3

train_path = 'G:/PyCharm/Lunwen3/corel-1000/train'
test_path='G:/PyCharm/Lunwen3/corel-1000/test'
logs_path = 'G:/PyCharm/Lunwen3/logs2_Pca'

train_data = dataset.read_train_sets(train_path, img_size1, img_size2,classes, validation_size=validation_size)
test_data=dataset.read_test_sets(test_path, img_size1,img_size2, classes)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(train_data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(train_data.valid.labels)))
print("Number of files in Testing-set:\t{}".format(len(test_data.test.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size1, img_size2, num_channels], name='x')
keep_prob = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

filter_size_conv1 = 5
num_filters_conv1 = 6

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64


W1=np.load(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W1.npy")
W2=np.load(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W2.npy")
W3=np.load(r"G:\PyCharm\Lunwen3\Ws_PCA_corel\W3.npy")


fc_layer_size = img_size1 * img_size2


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def create_biases(size):
    return tf.Variable(tf.constant(0.01, shape=[size]))

##################################################################网络函数#################################################################
def z_score(W):
    W=np.array(W,dtype=np.float32)
    W=W.transpose((1,2,3,0))
    w=create_weights(W.shape)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rand_w = w.eval(session=sess)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                for l in range(W.shape[3]):
                    if abs(W[i, j, k, l]) >= 0.01:
                        W[i, j, k, l] = rand_w[i, j, k, l]
                    else:
                        W[i, j, k, l] = W[i, j, k, l]
    W = tf.convert_to_tensor(w)

    return W


def create_convolutional_layer(input, num_filters, weights):
    weights = z_score(weights)
    biases = create_biases(num_filters)
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

    return layer


def create_flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

    return layer

################################################################################################################################
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases

    layer = tf.nn.dropout(layer, keep_prob=keep_prob)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

####################################################################################################################################

layer_conv1 = create_convolutional_layer(input=x, num_filters=num_filters_conv1,weights=W1)  # 这是一个计算过程

tf.summary.histogram('layer_conv1', layer_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,num_filters=num_filters_conv2,weights=W2)

tf.summary.histogram('layer_conv2', layer_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2, num_filters=num_filters_conv3,weights=W3)

tf.summary.histogram('layer_conv3', layer_conv3)

layer_flat = create_flatten_layer(layer_conv3)  # 拉长

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
    y_pred = tf.nn.softmax(layer_fc3, name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                               labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(cost)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
# 记录张量的数据
tf.summary.image('preprocess', tf.reshape(x, [-1, 32, 32, 3]), 10)
tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()  # 定义记录运算
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # 创建写对象


def show_progress(epoch, feed_dict_train, feed_dict_validate,feed_dict_test, val_loss, test_loss , i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    test_acc= session.run(accuracy, feed_dict=feed_dict_test)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}, Testing Accuracy: {5:>6.1%},  Testing Loss: {6:.3f}"
    print(msg.format(epoch + 1, i, acc, val_acc, val_loss,test_acc,test_loss))


total_iterations = 0  # 指定初始迭代次数

saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations  # 保证更新全局变量.

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_train_batch, y_train_batch, _, train_cls_batch = train_data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = train_data.valid.next_batch(batch_size)
        x_test=test_data.test.images
        y_test = test_data.test.labels

        feed_dict_tr = {x: x_train_batch,
                        y_true: y_train_batch,
                        keep_prob: 1.0}

        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch,
                         keep_prob: 1.0}

        feed_dict_test={x: x_test,
                        y_true: y_test,
                        keep_prob: 1.0}

        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(train_data.train.num_examples / batch_size) == 0:
            _, loss_, summary = session.run([optimizer, cost, merged_summary_op],feed_dict=feed_dict_val)  # 执行记录运算
            _, loss_, summary = session.run([optimizer, cost, merged_summary_op], feed_dict=feed_dict_test)
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            test_loss=session.run(cost, feed_dict=feed_dict_test)

            epoch = int(i / int(train_data.train.num_examples / batch_size))  # data.train.num_examples=800
            summary_writer.add_summary(summary, i)  # 将日志写入文件
            show_progress(epoch, feed_dict_tr, feed_dict_val, feed_dict_test, val_loss, test_loss, i)
            saver.save(session, './corel-Pca-classical-model/save_model.pd', global_step=i)  # 保存模型

    total_iterations += num_iteration


train(num_iteration=301)