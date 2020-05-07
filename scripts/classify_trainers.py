"""
"""
# Copyright (c) Cameron Fabbri
# All rights reserved
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn import svm

import utils.data_ops as do


class VGG(tf.keras.Model):

    def __init__(self):

        super(VGG, self).__init__()

        self.vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        self.vgg.trainable = False

        self.vgg_layers = []
        for layer in self.vgg.layers:
            self.vgg_layers.append(layer)
        self.vgg_layers = self.vgg_layers[:-1]

    def call(self, inputs):

        x = inputs
        for layer in self.vgg_layers:
            x = layer(x)
        return x


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Main data directory
    data_dir = os.path.join('data', 'sprites')

    pokemon_dir = os.path.join(data_dir, 'pokemon')
    pokemon_paths = do.get_paths(pokemon_dir)
    pokemon_paths.extend(do.get_paths(os.path.join(data_dir, 'pokemon_flat')))

    trainers_dir = os.path.join(data_dir, 'trainers')
    delete_dir = os.path.join(data_dir, 'delete')
    items_dir = os.path.join(data_dir, 'items')

    other_paths = []
    other_paths.extend(do.get_paths(trainers_dir))
    other_paths.extend(do.get_paths(os.path.join(data_dir, 'trainer_class', 'trainers')))
    other_paths.extend(do.get_paths(os.path.join(data_dir, 'random_sprites')))
    other_paths.extend(do.get_paths(os.path.join(data_dir, 'gyms')))
    other_paths.extend(do.get_paths(delete_dir))
    other_paths.extend(do.get_paths(items_dir))

    #pokemon_labels = np.ones((len(pokemon_paths)))
    #other_labels = np.zeros((len(other_paths)))

    vgg = VGG()

    def get_features(paths, label):
        train_features = np.empty((len(paths), 4096))
        train_labels = []
        i = 0
        for path in tqdm(paths):
            try:
                raw = tf.io.read_file(path)
                image = tf.io.decode_image(raw)[:,:,:3].numpy().astype(np.float32)
                image = np.expand_dims(image, 0)
                x = tf.keras.applications.vgg19.preprocess_input(image)
                x = tf.image.resize(x, (224, 224))
                feature = vgg(x).numpy()
                train_features[i, ...] = feature
                train_labels.append(label)
                i += 1
            except:
                print(path)
                continue
        return train_features, np.asarray(train_labels)

    print('Getting pokemon features...')
    pokemon_features, pokemon_labels = get_features(pokemon_paths, 1)

    print('Getting other features...')
    other_features, other_labels = get_features(other_paths, 0)

    features = np.concatenate([pokemon_features, other_features], axis=0)
    labels = np.concatenate([pokemon_labels, other_labels], axis=0)

    print(pokemon_features.shape)
    print(other_features.shape)
    print(features.shape,'\n')
    print(pokemon_labels.shape)
    print(other_labels.shape)
    print(labels.shape)

    print('\nFitting SVM')
    clf = svm.SVC()
    clf.fit(features, labels)
    print('Done')

    # Load testing data up
    testing_paths = [os.path.join(data_dir, 'trainer_class', 'other', x) for x in os.listdir(os.path.join(data_dir, 'trainer_class', 'other'))]
    testing_paths = [x for x in testing_paths if os.path.isfile(x)]

    new_pokemon_dir = os.path.join(data_dir, 'trainer_class', 'other', 'pokemon')

    for path in tqdm(testing_paths):

        try:
            raw = tf.io.read_file(path)
            image = tf.io.decode_image(raw)[:,:,:3].numpy().astype(np.float32)
            image = np.expand_dims(image, 0)
            x = tf.keras.applications.vgg19.preprocess_input(image)
            x = tf.image.resize(x, (224, 224))
            feature = vgg(x).numpy()
        except:
            print('Skipped',path)
            continue

        prediction = int(clf.predict(feature)[0])

        filename = os.path.basename(path)

        # Not a pokemon
        if prediction == 0:
            continue
        else:
            new_path = os.path.join(new_pokemon_dir, filename)
            os.rename(path, new_path)

if __name__ == '__main__':
    main()
