
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
import fixed_model

from PIL import Image, ImageDraw
import cPickle as pkl
from scipy.spatial import distance


def save_img_debug(strokes):
    image = Image.new("P", (128,128), color=255)
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=3)
    image.save('debug.png')


def strokes_to_img(strokes_):
    _tmp = np.concatenate(strokes_, axis=1)
    lower = np.min(_tmp, axis=1)
    strokes = []
    for s in strokes_:
        strokes.append(s-lower[:, None])

    image = Image.new("P", (128,128), color=255)
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=3)

    return np.array(image)

def draw_inc(img, stroke, sx, sy):
    image = Image.fromarray(img)
    image_draw = ImageDraw.Draw(image)

    for i in range(len(stroke[0])-1):
        image_draw.line([stroke[0][i]+sx,
                         stroke[1][i]+sy,
                         stroke[0][i+1]+sx,
                         stroke[1][i+1]+sy],
                        fill=0, width=3)

    return np.array(image)


class InferUtil(object):
    def __init__(self, model_ckpt, serve=False):

        utils.set_gpu(1)
        num_classes = 340
        image_size = 128
        self._is = image_size

        self.images = tf.placeholder(shape=[None, image_size, image_size, 1], dtype=tf.uint8)

        image_prep_fn = preprocessing_factory.get_preprocessing('inception_v1', is_training=False)
        images_preped = image_prep_fn(self.images, None, None)

        class_logits, self.feat = fixed_model.build_net(images_preped, num_classes, False)

        self.class_probs = tf.nn.softmax(class_logits)
        #preds = tf.argmax(class_probs, axis=-1)

        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        saver.restore(self.sess, model_ckpt)

        self.cls_id_map = pkl.load(file('class_id_map.pkl', 'rb'))
        self.id_cls_map = {}
        for k in self.cls_id_map:
            self.id_cls_map[self.cls_id_map[k]] = k

        if serve:
            self.vis = pkl.load(file('../../val_image_strokes.pkl', 'rb'))
            self.serve_hint()
            #self.serve_play()

    def serve_play(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % 10116)
        while True:
            z = socket.recv(flags=0)
            p = zlib.decompress(z)
            strokes = pkl.loads(p)
            img = strokes_to_img(strokes)

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


    def ext(self, image):
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
        image = np.asarray(image)
        bs = image.size / self._is / self._is
        if image.ndim in (2,3):
            image = np.reshape(image, [bs, self._is, self._is, 1])

        feats, probs = self.sess.run([self.feat, self.class_probs], {self.images: image})

        return feats, probs


    def serve_hint(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % 10116)

        target = 'flamingo'
        #target = 'chandelier'
        #target = 'apple'

        print 'start serving...'
        while True:
            #  Wait for next request from client
            z = socket.recv(flags=0)
            p = zlib.decompress(z)
            strokes = pkl.loads(p)
            img = strokes_to_img(strokes)

            print 'Top probs for incoming image:'
            print  self.infer(img, 5)
            #self.search(strokes, target)
            ret = self.judge(strokes, target)

            p = pkl.dumps(ret, -1)
            z = zlib.compress(p)
            socket.send(z, flags=0)


    def judge(self, strokes, target):
        if len(strokes) <= 1:
            return strokes
        img = strokes_to_img(strokes[:-1])
        target_id = self.cls_id_map[target]
        feats, probs = self.ext(img)
        prev_prob = probs[0][target_id]

        img = strokes_to_img(strokes)
        target_id = self.cls_id_map[target]
        feats, probs = self.ext(img)
        cur_prob = probs[0][target_id]
        print target, prev_prob, cur_prob,

        if cur_prob > prev_prob:
            print 'better'
            return strokes
        else:
            print 'roll back'
            return strokes[:-1]


    def search(self, strokes, target):
        img = strokes_to_img(strokes)
        target_id = self.cls_id_map[target]
        feats, probs = self.ext(img)
        init_prob = probs[0][target_id]
        print init_prob


        # find closest image of target class in val set, and get its strokes
        f = feats[0]
        md = 1e4
        mst = []
        for fv, st in self.vis[target]:
            d = distance.cosine(f, fv)
            if d < md:
                md = d
                mst = st


        all_imgs = []
        all_sts = []
        for s in mst: # each stroke
            if len(s) > 10:
                continue
            #for sx in range(0, 128, 16):
            #    for sy in range(0, 128, 16):
            for i in range(256):
                sx = random.randint(0,127)
                sy = random.randint(0,127)
                #new_img = draw_inc(img, s, sx, sy)

                new_img = strokes_to_img(strokes + [s + np.array([sx, sy])[:, None]])
                all_imgs.append(new_img)

        print 'searching:', len(all_imgs)
        _, probs = self.ext(all_imgs)
        max_p = 0.
        max_i = -1
        for i, p in enumerate(probs[:, target_id]):
            if p > max_p:
                max_p = p
                max_i = i
        Image.fromarray(all_imgs[max_i]).save('debug.png')
        print 'achieving prob:', max_p
        #save_img_debug(mst)






if __name__ == "__main__":
    iu = InferUtil('./ckpt/model-99001', True)

    #img = np.array(Image.open('bee.png'))

    #print img.shape

    #print iu.infer(img, 5)

