"""

Script to train an SVM to classify pokemon

"""
# Copyright (c) Cameron Fabbri
# All rights reserved
import os
import pickle
import random
import argparse

import cv2
import numpy as np

from tqdm import tqdm
from sklearn import svm

from pokemon_data import PokemonData


def main():

    seed_value = 3
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    args = parse()

    pd = PokemonData(args.data_dir)

    # 251 classes

    clf_file = 'pokemon_svm.pkl'
    train_images = []
    train_labels = []

    size = 64

    for pid, paths in tqdm(pd.pokemon.items()):
        for path in paths:
            image = cv2.imread(path)
            image = cv2.resize(image, (size, size)).flatten()
            train_images.append(image)
            train_labels.append(pid)

    # Randomly shuffle
    c = list(zip(train_images, train_labels))
    random.shuffle(c)
    train_images, train_labels = zip(*c)

    test_images = train_images[int(0.95*len(train_images)):]
    test_labels = train_labels[int(0.95*len(train_labels)):]

    train_images = train_images[:int(0.95*len(train_images))]
    train_labels = train_labels[:int(0.95*len(train_labels))]

    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    if not os.path.exists(clf_file):
        print('\nFitting SVM')
        clf = svm.SVC()
        clf.fit(train_images, train_labels)
        print('Done')

        with open('pokemon_svm.pkl', 'wb') as f:
            pickle.dump(clf, f)

    else:
        with open(clf_file, 'rb') as f:
            clf = pickle.load(f)

    total = 0
    correct = 0
    for test_image, test_label in zip(test_images, test_labels):

        test_image = np.reshape(test_image, (1, size*size))
        prediction = int(clf.predict(test_image)[0])
        print(prediction, test_label)

        if prediction == test_label:
            correct += 1
        total += 1

    print(float(correct)/float(total))


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--data_dir', type=str, required=True,
            help='')

    return parser.parse_args()

if __name__ == '__main__':
    main()
