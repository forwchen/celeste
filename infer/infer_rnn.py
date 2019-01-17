# coding: utf-8

import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

from PIL import Image
import cPickle as pkl

class InferUtilSketchRNN(object):
    def __init__(self, model_path = './ckpt/flamingo/lstm_uncond'):
        [self.hps_model, self.eval_hps_model, self.sample_hps_model] = self.load_model_compatible(model_path)
        
        # construct the sketch-rnn model here:
        reset_graph()
        self.model = Model(self.hps_model)
        self.eval_model = Model(self.eval_hps_model, reuse=True)
        self.sample_model = Model(self.sample_hps_model, reuse=True)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        # loads the weights from checkpoint into our model
        load_checkpoint(self.sess, model_path)
        

    def load_env_compatible(self, data_dir, model_dir):
        """Loads environment for inference mode, used in jupyter notebook."""
        # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
        # to work with depreciated tf.HParams functionality
        model_params = sketch_rnn_model.get_default_hparams()
        with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            data = json.load(f)
        fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
        for fix in fix_list:
            data[fix] = (data[fix] == 1)
        model_params.parse_json(json.dumps(data))
        return load_dataset(data_dir, model_params, inference_mode=True)

    def load_model_compatible(self, model_dir):
        """Loads model for inference mode, used in jupyter notebook."""
        # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
        # to work with depreciated tf.HParams functionality
        model_params = sketch_rnn_model.get_default_hparams()
        with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            data = json.load(f)
        fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
        for fix in fix_list:
            data[fix] = (data[fix] == 1)
        model_params.parse_json(json.dumps(data))

        model_params.batch_size = 1  # only sample one at a time
        eval_model_params = sketch_rnn_model.copy_hparams(model_params)
        eval_model_params.use_input_dropout = 0
        eval_model_params.use_recurrent_dropout = 0
        eval_model_params.use_output_dropout = 0
        eval_model_params.is_training = 0
        sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
        sample_model_params.max_seq_len = 1  # sample one point at a time
        return [model_params, eval_model_params, sample_model_params]

    def sample(self, sess, model, strokes_in=[], seq_len=250, temperature=1.0, greedy_mode=False, z=None):
        """Samples a sequence from a pre-trained model."""

        def adjust_temp(pi_pdf, temp):
            pi_pdf = np.log(pi_pdf) / temp
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        def get_pi_idx(x, pdf, temp=1.0, greedy=False):
            """Samples from a pdf, optionally greedily."""
            if greedy:
                return np.argmax(pdf)
            pdf = adjust_temp(np.copy(pdf), temp)
            accumulate = 0
            for i in range(0, pdf.size):
                accumulate += pdf[i]
                if accumulate >= x:
                    return i
            tf.logging.info('Error with sampling ensemble.')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
            if greedy:
                return mu1, mu2
            mean = [mu1, mu2]
            s1 *= temp * temp
            s2 *= temp * temp
            cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        if z is None:
            z = np.random.randn(1, model.hps.z_size)  # not used if unconditional
        if not model.hps.conditional:
            prev_state = sess.run(model.initial_state)
        else:
            prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

        strokes = np.zeros((seq_len, 5), dtype=np.float32)
        mixture_params = []

        greedy = greedy_mode
        temp = temperature

        for i in range(len(strokes_in)):
            if not model.hps.conditional:
                feed = {
                    model.input_x: prev_x,
                    model.sequence_lengths: [1],
                    model.initial_state: prev_state
                }
            else:
                feed = {
                    model.input_x: prev_x,
                    model.sequence_lengths: [1],
                    model.initial_state: prev_state,
                    model.batch_z: z
                }
            params = sess.run([
                model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
                model.pen, model.final_state
            ], feed)

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

            idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)
            idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
            eos = [0, 0, 0]
            eos[idx_eos] = 1

            strokes[i, :] = strokes_in[i, :]

            params = [
                o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
                o_pen[0]
            ]
            mixture_params.append(params)
            prev_x = np.zeros((1, 1, 5), dtype=np.float32)
            prev_x[0][0] = np.array(strokes_in[i, :], dtype=np.float32)
            prev_state = next_state

        for i in range(len(strokes_in), seq_len):
            if not model.hps.conditional:
                feed = {
                    model.input_x: prev_x,
                    model.sequence_lengths: [1],
                    model.initial_state: prev_state
                }
            else:
                feed = {
                    model.input_x: prev_x,
                    model.sequence_lengths: [1],
                    model.initial_state: prev_state,
                    model.batch_z: z
                }

            params = sess.run([
                model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
                model.pen, model.final_state
            ], feed)

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params
            idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)
            idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
            eos = [0, 0, 0]
            eos[idx_eos] = 1

            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                  o_sigma1[0][idx], o_sigma2[0][idx],
                                  o_corr[0][idx], np.sqrt(temp), greedy)
            strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

            params = [
                o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
                o_pen[0]
            ]
            mixture_params.append(params)

            prev_x = np.zeros((1, 1, 5), dtype=np.float32)
            prev_x[0][0] = np.array(
                [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
            prev_state = next_state
        
        return strokes, mixture_params
    
    def strokes_to_lines(self, strokes):
        """Convert stroke-3 format to polyline format."""
        x = 0
        y = 0
        lines = []
        line = []
        for i in range(len(strokes)):
            if strokes[i, 2] >= 1:
                x += float(strokes[i, 0])
                y += float(strokes[i, 1])
                line.append([x, y])
                lines.append(line)
                line = []
            else:
                x += float(strokes[i, 0])
                y += float(strokes[i, 1])
                line.append([x, y])
        return lines
        
    def to_normal_strokes(big_stroke):
        """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
        l = 0
        for i in range(len(big_stroke)):
            if big_stroke[i, 4] > 0:
                l = i
                break
        if l == 0:
            l = len(big_stroke)
        result = np.zeros((l, 3))
        result[:, 0:2] = big_stroke[0:l, 0:2]
        result[:, 2] = big_stroke[0:l, 3]
        return result
        
    def predict(self, lines, t=0.8):
        if(len(lines) < 1):
            return []
        # import ipdb; ipdb.set_trace()
        strokes_in = lines_to_strokes(map(lambda x: np.array(x)/20.0, lines))
        x0 = strokes_in[0][0]
        y0 = strokes_in[0][1]
        strokes_in[0] = np.array([0,0,0])
        sample_strokes, m = self.sample(self.sess, self.sample_model, to_big_strokes(strokes_in, max_len=len(strokes_in)), temperature=t)
        # import ipdb; ipdb.set_trace()
        sample_strokes[0][0] = x0
        sample_strokes[0][1] = y0
        strokes = to_normal_strokes(sample_strokes)
        if(len(strokes) == len(strokes_in)):
            print('Model End')
        strokes[:, 0:2] /= 0.05
        return self.strokes_to_lines(strokes)

if __name__ == "__main__":
    sketch_rnn = InferUtilSketchRNN()

    sketch_rnn.test()
    #img = np.array(Image.open('bee.png'))

    #print img.shape

    #print iu.infer(img, 5)

