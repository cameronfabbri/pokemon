"""

"""
#
#
import tensorflow as tf


def gan_G(d_fake):
    return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_fake), logits=d_fake))

def gan_D(d_real, d_fake):

    errD_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real, labels=tf.ones_like(d_real))
    errD_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake, labels=tf.zeros_like(d_fake))
    return tf.reduce_mean(errD_real + errD_fake)


@tf.function
def r1_penalty(d_out, x, gamma):

    gradients = tf.gradients(d_out, [x])[0]
    r1_penalty = (gamma / 2) * tf.square(gradients)
    return tf.reduce_mean(r1_penalty)
