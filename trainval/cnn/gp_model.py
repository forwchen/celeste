import tensorflow as tf

# global pooling model for cnn activation map visualization
def build_net(inputs, num_classes, training):
    net = tf.layers.conv2d(inputs, 64, (7,7), strides=(2,2), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (5,5), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 256, (3,3), padding='same', activation=tf.nn.relu)
    pre_pool = net
    net = tf.reduce_mean(net, [1, 2], name='global_pool')
    net = tf.layers.dropout(net, rate=0.5, training=training)
    net = tf.layers.dense(net, num_classes, activation=None)

    return pre_pool, net
