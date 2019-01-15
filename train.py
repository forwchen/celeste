
# coding: utf-8

# In[1]:


import os
import math
import utils
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime

from preprocessing import preprocessing_factory
slim = tf.contrib.slim
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lr', type=float, help='init learning rate')

args = parser.parse_args()

# In[3]:
utils.set_gpu(1)

# tunable
init_learning_rate = args.lr
learning_rate_decay_factor = 0.5
num_epochs_per_decay = 2
weight_decay = 0.00004

# fixed
batch_size = 512
num_classes = 340
image_size = 128
num_samples_per_epoch = num_classes * 10000


# In[4]:


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
        })
    image = tf.image.decode_png(features['image_raw'], channels=1)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [image_size, image_size, 1])

    return image, label

feat_dir = '/home/forwchen/daily/190114/tfrecords'
files = tf.data.Dataset.list_files(feat_dir+'/train-*.tfrecord').shuffle(100, seed=1234)

ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=8))
ds = ds.map(decode, num_parallel_calls=16)
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*10, seed=1234))
ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
ds = ds.prefetch(buffer_size=batch_size * 4)

iterator = tf.data.Iterator.from_structure(ds.output_types,
                                           ds.output_shapes)
images, labels = iterator.get_next()

training_init_op = iterator.make_initializer(ds)


# In[5]:


image_prep_fn = preprocessing_factory.get_preprocessing('inception_v1', is_training=False)
images_preped = image_prep_fn(images, None, None)
print images, images_preped

import model
class_logits = model.build_net(images_preped, num_classes, True)


labels_oh = tf.one_hot(labels, num_classes, on_value=1., off_value=0., dtype=tf.float32)

cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_oh, logits=class_logits)
cls_loss = tf.reduce_mean(cls_loss)

reg_loss = 0.#tf.add_n(tf.losses.get_regularization_losses())

tot_loss = cls_loss + reg_loss

tf.summary.scalar('loss/tot', tot_loss)
tf.summary.scalar('loss/cls', cls_loss)
tf.summary.scalar('loss/reg', reg_loss)


# In[6]:


global_step = tf.Variable(0, trainable=False, name='global_step')

decay_steps = num_samples_per_epoch * num_epochs_per_decay / batch_size

learning_rate = tf.train.exponential_decay(init_learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(tot_loss, global_step)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

tf.summary.scalar('lr', learning_rate)


# In[7]:


sess = tf.Session()
sess.run(training_init_op)
sess.run(tf.global_variables_initializer())


# In[8]:


train_dir = './log'
model_id = datetime.now().strftime("%H%M%S")
train_dir = os.path.join(train_dir, datetime.now().strftime("%Y%m%d"), model_id)
if not os.path.exists(train_dir): os.makedirs(train_dir)

summary_writer = tf.summary.FileWriter(train_dir, flush_secs=30)
merged_summ = tf.summary.merge_all()

for iter_ in tqdm(range(100000)):
    fet = sess.run([merged_summ, global_step, train_op])
    summary_writer.add_summary(fet[0], fet[1])
    #print fet[-1]
    if iter_ % 1000 == 0:
        saver.save(sess, os.path.join(train_dir, 'model'), global_step)

