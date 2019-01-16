
# coding: utf-8

# In[1]:

import sys
sys.path.append('../common')
import os
import sys
import math
import utils
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import cPickle as pkl
import ipdb

# In[2]:


from preprocessing import preprocessing_factory
slim = tf.contrib.slim

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ckpt', type=str, help='model checkpoint')

args = parser.parse_args()

# In[3]:
utils.set_gpu(1)

num_classes = 340
image_size = 128
batch_size = 100

num_samples = num_classes * 1000


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
files = tf.data.Dataset.list_files(feat_dir+'/val-*.tfrecord').shuffle(100, seed=1234)

ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=8))
ds = ds.map(decode, num_parallel_calls=16)
ds = ds.repeat()
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

import fixed_model
class_logits, feats = fixed_model.build_net(images_preped, num_classes, False)

class_probs = tf.nn.softmax(class_logits)

preds = tf.argmax(class_probs, axis=-1)


# In[6]:


saver = tf.train.Saver(tf.global_variables())


# In[7]:


sess = tf.Session()
sess.run(training_init_op)


# In[22]:


saver.restore(sess, args.ckpt)


# In[23]:

features = {}
cls_id = pkl.load(file('class_id_map.pkl', 'rb'))
id_cls = {}
for k in cls_id:
    id_cls[cls_id[k]] = k
    features[k] = []

for iter_ in tqdm(range(num_samples / batch_size), ncols=64):
    fet = sess.run([labels, preds, feats])

    for l, f in zip(fet[0], fet[2]):
        features[id_cls[l]].append(f)


pkl.dump(features, file('feats_all.pkl', 'wb'), 2)

