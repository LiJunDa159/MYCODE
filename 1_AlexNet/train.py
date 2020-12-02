import dataset
import tensorflow as tf
from alexnet import AlexNet
from numpy.random import seed   #随机种子，对数据切分，保证测试与训练集不随调试变化而变化
seed(10)  #10貌似像名字，设成什么都行
from tensorflow import set_random_seed
set_random_seed(20)

##########################参数区#############################
batch_size = 64
classes = ['0','1', '2','3','4','5','6','7','8','9']
num_classes = len(classes)
validation_size = 0.2 #如何切分
img_size = 227     #原始图形学32*32根本跑不出网络因此改成227，嘿嘿
num_channels = 3
train_path='G:\\MYWORK\\cifar\\train_cifar10'
logs_path='G:\\MYWORK\\1_AlexNet\\logs'


dropout_rate = 1.0
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 20

##############################数据load##########################
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
keep_prob = tf.placeholder(tf.float32)
################################网络load###############################################
model = AlexNet(x, dropout_rate, num_classes, train_layers)
with tf.name_scope('Y_pred'):
    y_pred = model.fc8
    y_pred_cls = tf.argmax(y_pred, dimension=1)
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,labels=y_true))
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())
##############################记录张量的数据,方便可视化##################################################
tf.summary.image('preprocess', tf.reshape(x, [-1, 227, 227, 3]), 10)
tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()  # 定义记录运算
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # 创建写对象
#################################################输出函数#######################################################
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1, i, acc, val_acc, val_loss))

################################迭代函数##############################################
total_iterations = 0  # 指定初始迭代次数
saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations  # 保证更新全局变量.

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples / batch_size) == 0:
            _, loss_, summary = session.run([optimizer, cost, merged_summary_op],
                                            feed_dict=feed_dict_val)  # 执行记录运算
            val_loss = session.run(cost, feed_dict=feed_dict_val)

            epoch = int(i / int(data.train.num_examples / batch_size))  # data.train.num_examples=800
            summary_writer.add_summary(summary, i)  # 将日志写入文件
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, i)
            saver.save(session, './AlexNet-model/save_model.pd', global_step=i)  # 保存模型

    total_iterations += num_iteration
#######################################正式训练########################################
train(num_iteration=1000)
####################################启动可视化###################################
# #cd G:\MYWORK\1_AlexNet
# G:
# activate tensorflow
# tensorboard --logdir=logs --host=127.0.0.1
# http://127.0.0.1:6006