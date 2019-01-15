import tensorflow as tf



def build_net(inputs, num_classes, training):

    net = tf.layers.conv2d(inputs, 32, (7,7), strides=(2,2), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 64, (5,5), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 64, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.2, training=training)
    net = tf.layers.dense(net, 512, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    net = tf.layers.dense(net, num_classes, activation=None)

    return net
