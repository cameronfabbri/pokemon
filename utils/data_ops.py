"""
"""
import os
import fnmatch

import cv2
import numpy as np

from PIL import Image, ImageOps


def normalize(x):
    return (x / 127.5) - 1.0


def unnormalize(x):
    return (x + 1.0) * 127.5


def save_image(f, x):
    try:
        x = x.numpy()
    except:
        pass
    cv2.imwrite(f, np.squeeze(unnormalize(x).astype(np.uint8)))


def crop_image(im):
    """ Crops image tight to the pokemon """

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
