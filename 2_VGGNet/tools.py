import tensorflow as tf

def conv(layer_name, x, out_channels, kernel_size=None, stride=None, is_pretrain=True):
    """
    Convolution op wrapper, the Activation id ReLU
    :param layer_name: layer name, eg: conv1, conv2, ...
    :param x: input tensor, size = [batch_size, height, weight, channels]
    :param out_channels: number of output channel (convolution kernel)
    :param kernel_size: convolution kernel size, VGG use [3,3]
    :param stride: paper default = [1,1,1,1]
    :param is_pretrain: whether you need pre train, if you get parameter from other, you don not want to train again,
                        so trainable = false. if not trainable = true
    :return: 4D tensor
    """
    kernel_size = kernel_size if kernel_size else [3, 3]
    stride = stride if stride else [1, 1, 1, 1]

    # x = tf.convert_to_tensor(x)
    in_channels =int(x.get_shape()[-1])

    with tf.variable_scope(layer_name):
        w = tf.get_variable(name="weights",
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=is_pretrain)
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            trainable=is_pretrain)
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x


def pool(layer_name, x, ksize=None, stride=None, is_max_pool=True):
    """
    Pooling op
    :param layer_name: layer name, eg:pool1, pool2,...
    :param x:input tensor
    :param ksize:pool kernel size, VGG paper use [1,2,2,1], the size of 2X2
    :param stride:stride size, VGG paper use [1,2,2,1]
    :param is_max_pool: default use max pool, if it is false, the we will use avg_pool
    :return: tensor
    """
    ksize = ksize if ksize else [1, 2, 2, 1]
    stride = stride if stride else [1, 2, 2, 1]

    if is_max_pool:
        x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

    return x


def batch_norm(x):
    """
    Batch Normalization (offset and scale is none). BN algorithm can improve train speed heavily.
    :param x: input tensor
    :return: norm tensor
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)

    return x


def FC_layer(layer_name, x, out_nodes):
    """
    Wrapper for fully-connected layer with ReLU activation function
    :param layer_name: FC layer name, eg: 'FC1', 'FC2', ...
    :param x: input tensor
    :param out_nodes: number of neurons for FC layer
    :return: tensor
    """
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # flatten into 1D
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)

        return x









