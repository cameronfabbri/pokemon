"""
"""
import os
import random
import fnmatch

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

from pokemon_data import PokemonData


def normalize(x):
    return (x / 127.5) - 1.0


def unnormalize(x):
    x = ((x+1.) / 2) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def to_image(x):
    """
    Converts input tensor to an image

    Tensor --> Numpy
    (bs, h, w, c) --> (h, w, c)
    Clip to range [-1, 1]
    [-1, 1] --> [0, 255]
    float32 --> uint8
    """

    try: x = x.numpy()
    except: pass
    if len(x.shape) > 3:
        x = x[0]
    x = unnormalize(x)
    x = x.astype(np.uint8)
    return x


def load_image(path):
    """
    Loads an image. If there is an alpha channel, it is converted to a white
    background
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return crop_sprite(image)


def crop_sprite(im):
    """
    Crops image tight to the pokemon

    This checks for the first row and column from all sides of the image that
    is not all white, and crops to there plus one pixel for padding
    """

    l = len(im.shape)
    if l < 3:
        im = np.expand_dims(im, 2)
        im = np.concatenate([im, im, im], 2)
    h, w, c = im.shape

    im_n = im/255.
    comb_img = np.sum(im_n, axis=2)

    # Loop through rows
    for n, row in enumerate(comb_img):
        if np.sum(row) < 3.*w:
            start_y = n
            break
    for n, row in enumerate(reversed(comb_img)):
        if np.sum(row) < 3.*w:
            end_y = n
            break

    # Loop through columns
    for n, col in enumerate(comb_img.T):
        if np.sum(col) < 3.*h:
            start_x = n
            break
    for n, col in enumerate(reversed(comb_img.T)):
        if np.sum(col) < 3.*h:
            end_x = n
            break

    end_y = h-end_y
    end_x = w-end_x

    if start_y > 0:
        start_y = start_y - 1
    if end_y < h:
        end_y = end_y + 1
    if start_x > 0:
        start_x = start_x - 1
    if end_x < h:
        end_x = end_x + 1

    cropped = im[start_y:end_y, start_x:end_x, :]

    return cropped


def get_paths(data_dir):
    """ Returns a list of paths """

    lower_exts = ['png', 'jpg', 'jpeg']
    all_exts = []
    for ext in lower_exts:
        all_exts.append(ext.lower())
        all_exts.append(ext.upper())

    image_paths = []
    for ext in all_exts:
        pattern = '*.' + ext
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d, filename)
                    image_paths.append(fname_)
    return image_paths


def get_afhq():

    data_dir = '/home/cameron/Research/datasets/data/afhq'

    train_data_dict = {}
    test_data_dict = {}

    cat_train_paths = np.asarray(get_paths(os.path.join(data_dir, 'train', 'cat')))
    dog_train_paths = np.asarray(get_paths(os.path.join(data_dir, 'train', 'dog')))
    train_data_dict[1] = cat_train_paths
    train_data_dict[2] = dog_train_paths

    cat_test_paths = np.asarray(get_paths(os.path.join(data_dir, 'val', 'cat')))
    dog_test_paths = np.asarray(get_paths(os.path.join(data_dir, 'val', 'dog')))
    test_data_dict[1] = cat_test_paths
    test_data_dict[2] = dog_test_paths

    return train_data_dict, test_data_dict


def get_pokemon_data(gens):

    data_dir = os.path.join('data','pokemon','done')
    pd = PokemonData(data_dir)

    train_data_dict = {}
    test_data_dict = {}

    if 1 in gens:
        gen1_paths = pd.get_paths_from_gen(1)
        random.shuffle(gen1_paths)
        gen1_train_paths = np.asarray(gen1_paths[:int(0.95*len(gen1_paths))])
        gen1_test_paths = np.asarray(gen1_paths[int(0.95*len(gen1_paths)):])
        train_data_dict[1] = gen1_train_paths
        test_data_dict[1] = gen1_test_paths

    if 2 in gens:
        gen2_paths = pd.get_paths_from_gen(2)
        random.shuffle(gen2_paths)
        gen2_train_paths = np.asarray(gen2_paths[:int(0.95*len(gen2_paths))])
        gen2_test_paths = np.asarray(gen2_paths[int(0.95*len(gen2_paths)):])
        train_data_dict[2] = gen2_train_paths
        test_data_dict[2] = gen2_test_paths

    if 4 in gens:
        gen4_paths = pd.get_paths_from_gen(4)
        random.shuffle(gen4_paths)
        gen4_train_paths = np.asarray(gen4_paths[:int(0.95*len(gen4_paths))])
        gen4_test_paths = np.asarray(gen4_paths[int(0.95*len(gen4_paths)):])
        train_data_dict[4] = gen4_train_paths
        test_data_dict[4] = gen4_test_paths

    if 5 in gens:
        gen5_paths = pd.get_paths_from_gen(5)
        random.shuffle(gen5_paths)
        gen5_train_paths = np.asarray(gen5_paths[:int(0.95*len(gen5_paths))])
        gen5_test_paths = np.asarray(gen5_paths[int(0.95*len(gen5_paths)):])
        train_data_dict[5] = gen5_train_paths
        test_data_dict[5] = gen5_test_paths

    return train_data_dict, test_data_dict

