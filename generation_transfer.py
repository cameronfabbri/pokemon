"""

"""
#
#
import os
import random

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import network
import utils.tf_ops as tfo
import utils.data_ops as do

from pokemon_data import PokemonData


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 3
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    data_dir = os.path.join('data','pokemon','done')
    pd = PokemonData(data_dir)

    image = tf.random.normal((4, 64, 64, 3), dtype=tf.float32)

    style = tf.random.normal((4, 16), dtype=tf.float32)

    generator = network.Generator()
    generator(image, style)

    discriminator = network.Discriminator()

    #gen1_paths = pd.get_paths_from_gen(1)
    #gen2_paths = pd.get_paths_from_gen(2)
    #gen4_paths = pd.get_paths_from_gen(4)
    #gen5_paths = pd.get_paths_from_gen(5)

    #cv2.imwrite('image.png', image)
    #do.crop_image(image)







if __name__ == '__main__':
    main()
