"""

"""
#
#
import os
import sys
import math

import tensorflow_addons as tfa

import tensorflow as tf
import utils.tf_ops as tfo


class AdaIn(tf.keras.Model):

    def __init__(
            self,
            dim_out):

        super(AdaIn, self).__init__()

        self.fc = tf.keras.layers.Dense(
                units=dim_out*2,
                kernel_initializer='ones')

    def call(self, inputs, style):

        # Normalize inputs
        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs) +1e-8
        y = (inputs - mean) / std

        # Put style through a fc layer
        s = self.fc(style)

        shape = int(s.shape[-1]/2)

        gamma = s[:, :shape]
        beta = s[:, shape:]

        # Split up the output from the fc layer into two parts
        gamma = tf.reshape(gamma, [-1, 1, 1, shape])
        beta = tf.reshape(beta, [-1, 1, 1, shape])

        return y * gamma + beta


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
            dim_in,
            dim_out,
            upsample):

        super(AdaInResBlock, self).__init__()

        self.upsample = upsample

        # True if the input and output filter sizes don't match up
        self.filter_match = dim_in != dim_out

        self.norm1 = AdaIn(dim_out=dim_in)
        self.norm2 = AdaIn(dim_out=dim_out)

        self.conv1 = tf.keras.layers.Conv2D(
                filters=dim_out,
                strides=1,
                kernel_size=3,
                padding='SAME',
                kernel_initializer='he_normal')

        self.conv2 = tf.keras.layers.Conv2D(
                filters=dim_out,
                strides=1,
                kernel_size=3,
                padding='SAME',
                kernel_initializer='he_normal')

        # Convolution to match up input and output filters
        if self.filter_match:
            self.conv_1x1 = tf.keras.layers.Conv2D(
                    filters=dim_out,
                    strides=1,
                    kernel_size=1,
                    padding='SAME',
                    use_bias=False,
                    kernel_initializer='he_normal')

        if self.upsample:
            self.upsample_layer_s = tf.keras.layers.UpSampling2D(size=(2,2))
            self.upsample_layer_r = tf.keras.layers.UpSampling2D(size=(2,2))


    def _shortcut(self, x):

        if self.upsample:
            x = self.upsample_layer_s(x)

        if self.filter_match:
            x = self.conv_1x1(x)

        return x

    def _residual(self, x, s):

        x = self.norm1(x, s)
        x = tf.nn.leaky_relu(x)

        if self.upsample:
            x = self.upsample_layer_r(x)

        x = self.conv1(x)
        x = self.norm2(x, s)
        x = tf.nn.leaky_relu(x)

        x = self.conv2(x)

        return x

    def call(self, inputs, style, training=True):

        x_s = self._shortcut(inputs)
        x_r = self._residual(inputs, style)

        x = x_s + x_r

        x = x / math.sqrt(2)

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
            dim_in,
            dim_out,
            downsample):

        super(ResBlock, self).__init__()

        self.downsample = downsample

        # True if the input and output filter sizes don't match up
        self.filter_match = dim_in != dim_out

        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
                filters=dim_out,
                kernel_size=3,
                padding='SAME',
                strides=1, kernel_initializer='he_normal')

        self.norm2 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
                filters=dim_out,
                kernel_size=3,
                padding='SAME',
                strides=1, kernel_initializer='he_normal')

        # Convolution to match up input and output filters
        if self.filter_match:
            self.conv_1x1 = tf.keras.layers.Conv2D(
                    filters=dim_out,
                    strides=1,
                    kernel_size=1,
                    padding='SAME',
                    use_bias=False,
                    kernel_initializer='he_normal')

        if self.downsample:
            self.avg_pool_r = tf.keras.layers.AveragePooling2D(pool_size=(2,2))
            self.avg_pool_s = tf.keras.layers.AveragePooling2D(pool_size=(2,2))


    def _shortcut(self, x):

        if self.filter_match:
            x = self.conv_1x1(x)

        if self.downsample:
            x = self.avg_pool_s(x)

        return x

    def _residual(self, x):

        x = self.norm1(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv1(x)

        if self.downsample:
            x = self.avg_pool_r(x)

        x = self.norm2(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        return x


    def call(self, inputs, training=True):

        x_s = self._shortcut(inputs)
        x_r = self._residual(inputs)

        x = x_s + x_r

        x = x / math.sqrt(2)

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
                kernel_initializer='he_normal')

        # Downsampling blocks
        self.block1 = ResBlock(dim_in=64, dim_out=128, downsample=True)
        self.block2 = ResBlock(dim_in=128, dim_out=256, downsample=True)
        self.block3 = ResBlock(dim_in=256, dim_out=512, downsample=True)
        self.block4 = ResBlock(dim_in=512, dim_out=512, downsample=True)

        # Intermediate blocks
        self.block5 = ResBlock(dim_in=512, dim_out=512, downsample=False)
        self.block6 = ResBlock(dim_in=512, dim_out=512, downsample=False)

        # Intermediate style blocks
        self.block7 = AdaInResBlock(dim_in=512, dim_out=512, upsample=False)
        self.block8 = AdaInResBlock(dim_in=512, dim_out=512, upsample=False)

        # Upsampling style blocks
        self.block9 = AdaInResBlock(dim_in=512, dim_out=512, upsample=True)
        self.block10 = AdaInResBlock(dim_in=512, dim_out=256, upsample=True)
        self.block11 = AdaInResBlock(dim_in=256, dim_out=128, upsample=True)
        self.block12 = AdaInResBlock(dim_in=128, dim_out=64, upsample=True)

        self.norm_out = tfa.layers.InstanceNormalization()
        self.conv_out = tf.keras.layers.Conv2D(
                filters=3,
                strides=1,
                kernel_size=1,
                padding='SAME')


    def call(self, inputs, style, training=True):

        x = self.conv1(inputs)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x, style)
        x = self.block8(x, style)
        x = self.block9(x, style)
        x = self.block10(x, style)
        x = self.block11(x, style)
        x = self.block12(x, style)

        x = tf.nn.leaky_relu(x)
        x = self.norm_out(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv_out(x)

        return x

class Discriminator(tf.keras.Model):

    def __init__(self, c_dim):
        super(Discriminator, self).__init__()

        self.c_dim = c_dim

        # Initial convolution
        self.conv1 = tf.keras.layers.Conv2D(
                filters=64,
                strides=1,
                kernel_size=3,
                padding='SAME',
                kernel_initializer='he_normal')

        self.block1 = ResBlock(dim_in=64, dim_out=128, downsample=True)
        self.block2 = ResBlock(dim_in=128, dim_out=256, downsample=True)
        self.block3 = ResBlock(dim_in=256, dim_out=512, downsample=True)
        self.block4 = ResBlock(dim_in=512, dim_out=512, downsample=True)

        # Final convolutions
        self.conv2 = tf.keras.layers.Conv2D(
                filters=512,
                strides=1,
                kernel_size=4,
                padding='VALID',
                kernel_initializer='he_normal')

        self.conv3 = tf.keras.layers.Conv2D(
                filters=c_dim,
                strides=1,
                kernel_size=1,
                padding='VALID',
                kernel_initializer='he_normal')

    def call(self, inputs, y_label, training=True):

        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)

        x = tf.reshape(x, [-1, self.c_dim])

        return tf.expand_dims(tf.gather_nd(x, y_label), 1)


class Encoder(tf.keras.Model):

    def __init__(
            self,
            c_dim,
            style_dim):

        super(Encoder, self).__init__()

        self.c_dim = c_dim
        self.style_dim = style_dim

        # Initial convolution
        self.conv1 = tf.keras.layers.Conv2D(
                filters=64,
                strides=1,
                kernel_size=3,
                padding='SAME',
                kernel_initializer='he_normal')

        self.block1 = ResBlock(dim_in=64, dim_out=128, downsample=True)
        self.block2 = ResBlock(dim_in=128, dim_out=256, downsample=True)
        self.block3 = ResBlock(dim_in=256, dim_out=512, downsample=True)
        self.block4 = ResBlock(dim_in=512, dim_out=512, downsample=True)
        self.block5 = ResBlock(dim_in=512, dim_out=512, downsample=True)
        self.block6 = ResBlock(dim_in=512, dim_out=512, downsample=True)


        # Final convolution
        self.conv2 = tf.keras.layers.Conv2D(
                filters=64,
                strides=1,
                kernel_size=4,
                padding='SAME',
                activation=tf.nn.leaky_relu, kernel_initializer='he_normal')

        fc_layers = []
        for i in range(c_dim):
            fc_layers.append(tf.keras.layers.Dense(self.style_dim, kernel_initializer='he_normal'))
        self.fc_layers = fc_layers

    def call(self, inputs, y_label, training=True):

        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = tf.reshape(x, [-1, 512])

        fc_layers = []
        for layer in self.fc_layers:
            fc_layers.append(layer(x))

        return tf.gather_nd(fc_layers, y_label)


class MappingBlock(tf.keras.Model):

    def __init__(
            self,
            style_dim):
        super(MappingBlock, self).__init__()

        self.layer1 = tf.keras.layers.Dense(
                units=512,
                activation=tf.nn.relu, kernel_initializer='he_normal')
        self.layer2 = tf.keras.layers.Dense(
                units=512,
                activation=tf.nn.relu, kernel_initializer='he_normal')
        self.layer3 = tf.keras.layers.Dense(
                units=512,
                activation=tf.nn.relu, kernel_initializer='he_normal')
        self.layer4 = tf.keras.layers.Dense(
                units=style_dim, kernel_initializer='he_normal')

    def call(self, inputs, training=True):

        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class MappingNetwork(tf.keras.Model):

    def __init__(
            self,
            c_dim,
            style_dim):
        super(MappingNetwork, self).__init__()

        self.c_dim = c_dim

        shared_layers = []
        class_blocks = []

        for i in range(4):
            shared_layers.append(
                    tf.keras.layers.Dense(
                        units=512,
                        activation=tf.nn.relu, kernel_initializer='he_normal'))

        for i in range(self.c_dim):
            class_blocks.append(MappingBlock(style_dim))

        self.shared_layers = shared_layers
        self.class_blocks = class_blocks

    def call(self, inputs, y_label, training=True):

        x = inputs
        for layer in self.shared_layers:
            x = layer(x)

        output_styles = []

        for block in self.class_blocks:
            output_styles.append(block(x))

        out = tf.gather_nd(output_styles, y_label)
        return out
