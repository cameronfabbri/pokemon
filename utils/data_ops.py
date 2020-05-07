"""
"""
import os
import fnmatch

import cv2
import numpy as np

from PIL import Image, ImageOps


def crop_image(im):
    """ Crops image tight to the pokemon """
    '''
    h, w, c = im.shape
    white_col = h*3*255
    white_row = w*3*255

    for col_num, col in enumerate(im):
        if np.sum(col) != white_col:
            col_start = col_num
            break

    for col_num, col in enumerate(np.fliplr(im)):
        if np.sum(col) != white_col:
            col_end = w - col_num
            break

    im = np.rot90(im)

    for row_num, row in enumerate(im):
        if np.sum(row) != white_row:
            row_start = row_num
            break

    for row_num, row in enumerate(np.fliplr(im)):
        if np.sum(row) != white_row:
            row_end = w - row_num
            break

    im = np.rot90(im)
    im = np.rot90(im)
    im = np.rot90(im)
    crop = im[col_start:col_end, row_start:row_end, :]
    '''

    padding = 1
    image=Image.open(im)
    image.load()
    imageSize = image.size

    # remove alpha channel
    invert_im = image.convert("RGB")

    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()
    imageBox = tuple(np.asarray(imageBox))#+padding)

    cropped = image.crop(imageBox)
    cropped.save('image2.png')
    #cv2.imwrite('image2.png', out)
    exit()
    return x

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
