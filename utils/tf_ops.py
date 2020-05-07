"""

"""
import tensorflow as tf


def ada_in(x, gamma, beta):
    """
    """

    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x) +1e-8

    y = (x - mean) / std
    gamma = tf.reshape(gamma, [-1, 1, 1, y.shape[-1]])
    beta = tf.reshape(beta, [-1, 1, 1, y.shape[-1]])

    return y * gamma + beta


def pixel_norm(x, epsilon=1e-8):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
