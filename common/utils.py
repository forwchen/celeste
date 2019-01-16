import os
import sys
import GPUtil
import operator
import tensorflow as tf

def print_file(f):
    for l in open(f, 'r').readlines():
        sys.stdout.write(l)

def write_file(s,n):
    with open(n, 'w') as f:
        f.write(s)

def cfg_dict2tuple(d):
    cfg = namedtuple('Config', d.keys())
    for k in d.keys():
        exec('cfg.%s = d[k]' % k)
    return cfg

def set_gpu(num_gpu):
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = num_gpu, maxMemory = 0.2)
    if len(deviceIDs) < num_gpu:
        print 'Using CPU.'
        return
    else:
        gpu_id = ','.join([str(d) for d in deviceIDs])

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


def combine_gradients(tower_grads):
    filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
    final_grads = []
    for i in xrange(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in xrange(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((grad, filtered_grads[0][i][1],))

    return final_grads


