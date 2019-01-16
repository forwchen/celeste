
# coding: utf-8

import os
import sys
import zmq
import zlib
import math
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
slim = tf.contrib.slim

sys.path.append('../common')
from preprocessing import preprocessing_factory
import utils
import model

from PIL import Image
import cPickle as pkl

class InferUtil(object):
    def __init__(self, model_ckpt, serve=False):

        utils.set_gpu(1)
        num_classes = 340
        image_size = 128
        self._is = image_size

        self.images = tf.placeholder(shape=[None, image_size, image_size, 1], dtype=tf.uint8)

        image_prep_fn = preprocessing_factory.get_preprocessing('inception_v1', is_training=False)
        images_preped = image_prep_fn(self.images, None, None)

        class_logits = model.build_net(images_preped, num_classes, False, 'light')

        self.class_probs = tf.nn.softmax(class_logits)
        #preds = tf.argmax(class_probs, axis=-1)

        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        saver.restore(self.sess, model_ckpt)

        cls_id_map = pkl.load(file('class_id_map.pkl', 'rb'))
        self.id_cls_map = {}
        for k in cls_id_map:
            self.id_cls_map[cls_id_map[k]] = k

        if serve:
            self.start_serve()

    def start_serve(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % 10116)
        while True:
            #  Wait for next request from client
            meta = socket.recv_json(flags=0)
            data = socket.recv(flags=0, copy=True, track=False)
            buf = buffer(data)
            img = np.frombuffer(buf, dtype=meta['dtype']).reshape(meta['shape'])
            ret = self.infer(img, 5)
            print ret
            p = pkl.dumps(ret, -1)
            z = zlib.compress(p)
            socket.send(z, flags=0)

    def infer(self, image, k):
        """
        Parameters
        ----------
        image : ndarray of uint8
            image can be 2D, 3D or 4D tensor.
            2D: [h, w]
            3D: [h, w, 1]
            4D: [b, h, w, 1]

        k : int
            top-k predictions to return

        Returns
        -------
        nested list of length 1 or b:
            returns a list for each image in the batch,
            in which items are (prob, class_name) tuples, sorted by prob in descending order.
        """

        if image.ndim in (2,3):
            image = np.reshape(image, [1, self._is, self._is, 1])

        probs = self.sess.run(self.class_probs, {self.images: image})

        ret = []
        for prob in probs:
            idxs = np.argsort(prob)[::-1][:k]
            ret.append([(prob[i], self.id_cls_map[i]) for i in idxs])
        return ret





if __name__ == "__main__":
    iu = InferUtil('./ckpt/model-99001', True)

    #img = np.array(Image.open('bee.png'))

    #print img.shape

    #print iu.infer(img, 5)

