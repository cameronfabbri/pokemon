"""

"""
#
#
import os
import sys

import tensorflow as tf
import tensorflow_addons as tfa

import utils.tf_ops as tfo


class AdaInResBlock(tf.keras.Model):
    """
    ReLU
    AdaIn
    Conv

    ReLU
    AdaIn
    Conv

    Conv on inputs to match filters
    Add

    Optional Upsample
    """

    def __init__(
            self,
            filters,
            gb1_dim,
            gb2_dim,
            upsample):
        """
        gb1_dim is the number of filters of the input
        gb2_dim is the number of filters of the output
        """
        super(AdaInResBlock, self).__init__()

        self.upsample = upsample

        # Gamma and Beta AdaIn fc layers
        self.fc_g1 = tf.keras.layers.Dense(gb1_dim)
        self.fc_b1 = tf.keras.layers.Dense(gb1_dim)
        self.fc_g2 = tf.keras.layers.Dense(gb2_dim)
        self.fc_b2 = tf.keras.layers.Dense(gb2_dim)

        self.conv1 = tf.keras.layers.Conv2D(
                filters=filters,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)

        self.conv2 = tf.keras.layers.Conv2D(
                filters=filters,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)

        self.conv_m = tf.keras.layers.Conv2D(
                filters=filters,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)

        if self.upsample:
            self.upsample_layer = tf.keras.layers.UpSampling2D(size=(2,2))

    def call(self, inputs, style):

        g1 = self.fc_g1(style)
        g2 = self.fc_g2(style)
        b1 = self.fc_b1(style)
        b2 = self.fc_b2(style)

        x = tf.nn.relu(inputs)
        x = tfo.ada_in(x, g1, b1)
        x = self.conv1(x)

        x = tf.nn.relu(x)
        x = tfo.ada_in(x, g2, b2)
        x = self.conv2(x)

        inputs = self.conv_m(inputs)
        x = x + inputs

        if self.upsample:
            x = self.upsample_layer(x)

        return x


class ResBlock(tf.keras.Model):
    """
    ReLU
    Instance Norm
    Conv

    ReLU
    Instance Norm
    Conv

    Conv on inputs to match filters
    Add

    Optional Downsample
    """
    def __init__(
            self,
            filters,
            downsample):

        super(ResBlock, self).__init__()

        self.downsample = downsample

        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                padding='SAME',
                strides=1)

        self.norm2 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                padding='SAME',
                strides=1)

        # Convolution to match up input and output filters
        self.conv_m = tf.keras.layers.Conv2D(
                filters=filters,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)

        if self.downsample:
            self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2,2))

    def call(self, inputs):

        x = tf.nn.relu(inputs)
        x = self.norm1(x)
        x = self.conv1(x)

        x = tf.nn.relu(x)
        x = self.norm2(x)
        x = self.conv2(x)

        inputs = self.conv_m(inputs)

        x = x + inputs

        if self.downsample:
            x = self.avg_pool(x)

        return x


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        # Initial convolution
        self.conv1 = tf.keras.layers.Conv2D(
                filters=64,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)

        # Downsampling blocks
        self.block1 = ResBlock(128, True)
        self.block2 = ResBlock(256, True)
        self.block3 = ResBlock(512, True)
        self.block4 = ResBlock(512, True)

        # Intermediate blocks
        self.block5 = ResBlock(512, False)
        self.block6 = ResBlock(512, False)

        # Intermediate style blocks
        self.block7 = AdaInResBlock(512, 512, 512, False)
        self.block8 = AdaInResBlock(512, 512, 512, False)

        # Upsampling style blocks
        self.block9 = AdaInResBlock(512, 512, 512, True)
        self.block10 = AdaInResBlock(256, 512, 256, True)
        self.block11 = AdaInResBlock(128, 256, 128, True)
        self.block12 = AdaInResBlock(64, 128, 64, True)

        self.conv_out = tf.keras.layers.Conv2D(
                filters=3,
                strides=1,
                kernel_size=3,
                padding='SAME',
                activation=None)


    def call(self, inputs, style):

        f = open(os.devnull, 'w')
        sys.stdout = f

        print('\n')
        print('Inputs')
        print('-------------------')
        print(inputs.shape,'\n')
        print('Conv 1')
        print('-------------------')
        x = self.conv1(inputs)
        print('x:',x.shape,'\n')

        print('Downsampling Blocks')
        print('-------------------')
        x = self.block1(x)
        print('x:',x.shape)
        x = self.block2(x)
        print('x:',x.shape)
        x = self.block3(x)
        print('x:',x.shape)
        x = self.block4(x)
        print('x:',x.shape,'\n')

        print('Intermediate Blocks:')
        print('-------------------')
        x = self.block5(x)
        print('x:',x.shape)
        x = self.block6(x)
        print('x:',x.shape,'\n')

        print('Intermediate Style Blocks:')
        print('-------------------')
        x = self.block7(x, style)
        print('x:',x.shape)
        x = self.block8(x, style)
        print('x:',x.shape,'\n')

        print('Upsampling Style Blocks:')
        print('-------------------')
        x = self.block9(x, style)
        print('x:',x.shape)
        x = self.block10(x, style)
        print('x:',x.shape)
        x = self.block11(x, style)
        print('x:',x.shape)
        x = self.block12(x, style)
        print('x:',x.shape,'\n')

        print('Conv Out')
        print('-------------------')
        x = self.conv_out(x)
        print('x:',x.shape)

        sys.stdout = sys.__stdout__
        return x
