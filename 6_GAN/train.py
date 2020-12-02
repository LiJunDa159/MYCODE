import dataset
import tensorflow as tf
import gan
import numpy as np
import pickle
import matplotlib.pyplot as plt
##########################参数区#############################
batch_size = 5
classes = ['0','1', '2','3','4','5','6','7','8','9']
num_classes = len(classes)
validation_size = 0.2 #如何切分
img_size = 32*32
noise_size = 100
num_channels = 3
g_units = 128
d_units = 128
learning_rate = 0.001
epochs = 50
n_sample = 5
alpha = 0.01
# train_path='G:\\MYWORK\\cifar\\train_cifar10'
train_path='G:\\MYWORK\\6_GAN\\train_data'
logs_path='G:\\MYWORK\\6_GAN\\logs'
##############################数据load##########################
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
session = tf.Session()
real_img, noise_img = gan.get_inputs(img_size, noise_size)
################################网络load###############################################
# generator
g_logits, g_outputs = gan.get_generator(noise_img, g_units, img_size)

# discriminator
d_logits_real, d_outputs_real = gan.get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = gan.get_discriminator(g_outputs, d_units, reuse=True)

# discriminator的loss
# 识别真实图片
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)))
# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real, d_loss_fake)

# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)))
train_vars = tf.trainable_variables()

# generator
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# 存储测试样例
samples = []
# 存储loss
losses = []
# 保存生成器变量
saver = tf.train.Saver(var_list=g_vars)
tf.reset_default_graph()
# ##########################################开始训练######################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(data.train.num_examples // batch_size):
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            x_batch = x_batch * 2 - 1
            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            x_batch=tf.reshape(x_batch,[-1,img_size])
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={real_img: x_batch, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss,
                                feed_dict={real_img: x_batch,
                                           noise_img: batch_noise})
        # real img loss
        train_loss_d_real = sess.run(d_loss_real,
                                     feed_dict={real_img: x_batch,
                                                noise_img: batch_noise})
        # fake img loss
        train_loss_d_fake = sess.run(d_loss_fake,
                                     feed_dict={real_img: x_batch,
                                                noise_img: batch_noise})
        # generator loss
        train_loss_g = sess.run(g_loss,
                                feed_dict={noise_img: batch_noise})

        print("Epoch {}/{}...".format(e + 1, epochs),
              "判别器损失: {:.4f}(判别真实的: {:.4f} + 判别生成的: {:.4f})...".format(train_loss_d, train_loss_d_real,
                                                                       train_loss_d_fake),
              "生成器损失: {:.4f}".format(train_loss_g))

        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        # 保存样本
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(gan.get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)

        saver.save(sess, './checkpoints/generator.ckpt')

# 保存到本地
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

fig, ax = plt.subplots(figsize=(20, 7))
losses = np.array(losses)
plt.plot(losses.T[0], label='判别器总损失')
plt.plot(losses.T[1], label='判别真实损失')
plt.plot(losses.T[2], label='判别生成损失')
plt.plot(losses.T[3], label='生成器损失')
plt.title("对抗生成网络")
ax.set_xlabel('epoch')
plt.legend()

# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

# samples是保存的结果 epoch是第多少次迭代
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):  # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

    return fig, axes

_ = view_samples(-1, samples)  # 显示最终的生成结果

# 指定要查看的轮次
epoch_idx = [10, 30, 60, 90, 120, 150, 180, 210, 240, 290]
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i][1])

rows, cols = 10, 25
fig, axes = plt.subplots(figsize=(30, 12), nrows=rows, ncols=cols)

idx = range(0, epochs, int(epochs / rows))

for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


# 加载我们的生成器变量
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
    gen_samples = sess.run(gan.get_generator(noise_img, g_units, img_size, reuse=True),
                           feed_dict={noise_img: sample_noise})

_ = view_samples(0, [gen_samples])


