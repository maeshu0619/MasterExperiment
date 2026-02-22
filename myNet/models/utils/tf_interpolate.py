import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.framework import ops
import sys
import os
import numpy as np
import time

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

# custom op のロード
interpolate_module = tf.load_op_library(
    os.path.join(BASE_DIR, 'tf_interpolate_so.so')
)

def three_nn(xyz1, xyz2):
    """
    xyz1: (B, N, 3) unknown points
    xyz2: (B, M, 3) known points
    """
    return interpolate_module.three_nn(xyz1, xyz2)

def three_interpolate(points, idx, weight):
    """
    points: (B, M, C)
    idx:    (B, N, 3)
    weight: (B, N, 3)
    """
    return interpolate_module.three_interpolate(points, idx, weight)

@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points, idx, weight = op.inputs
    return [
        interpolate_module.three_interpolate_grad(
            points, idx, weight, grad_out
        ),
        None,
        None
    ]

if __name__ == '__main__':
    np.random.seed(100)

    B = 32
    M = 128   # known points
    N = 512   # unknown points
    C = 64

    points_np = np.random.random((B, M, C)).astype(np.float32)
    xyz1_np = np.random.random((B, N, 3)).astype(np.float32)
    xyz2_np = np.random.random((B, M, 3)).astype(np.float32)

    with tf.device('/cpu:0'):
        points = tf.constant(points_np)
        xyz1 = tf.constant(xyz1_np)
        xyz2 = tf.constant(xyz2_np)

        dist, idx = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist) / 3.0
        interpolated_points = three_interpolate(points, idx, weight)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        now = time.time()
        for _ in range(100):
            ret = sess.run(interpolated_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
