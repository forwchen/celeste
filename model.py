import tensorflow as tf



def build_light(inputs, num_classes, training):

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

def build_wide(inputs, num_classes, training):

    net = tf.layers.conv2d(inputs, 64, (7,7), strides=(2,2), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (5,5), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.3, training=training)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    net = tf.layers.dense(net, num_classes, activation=None)

    return net

def build_deep(inputs, num_classes, training):

    net = tf.layers.conv2d(inputs, 32, (7,7), strides=(2,2), padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 32, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 64, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 64, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 64, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.3, training=training)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    net = tf.layers.dense(net, num_classes, activation=None)

    return net

def build_gp(inputs, num_classes, training):

    net = tf.layers.conv2d(inputs, 64, (7,7), strides=(2,2), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (5,5), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 128, (3,3), padding='same', activation=tf.nn.relu)
    net = tf.reduce_mean(net, [1, 2], name='global_pool')
    net = tf.layers.dropout(net, rate=0.3, training=training)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    net = tf.layers.dense(net, num_classes, activation=None)

    return net


def build_net(inputs, num_classes, training, model):
    if model == 'light':
        return build_light(inputs, num_classes, training)
    elif model == 'wide':
        return build_wide(inputs, num_classes, training)
    elif model == 'deep':
        return build_deep(inputs, num_classes, training)
    elif model == 'gp':
        return build_gp(inputs, num_classes, training)
