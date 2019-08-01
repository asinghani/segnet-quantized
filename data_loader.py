from sklearn.model_selection import train_test_split
import os, sys
import glob
import cv2
import numpy as np
import random
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from vis import img_stats

from augmentation import augment

INPUTSHAPE = 128

def load_data(config = [], aug_data = True):
    """
    Loads and prepares data. Returns generators for (train, test)
    """
    X_train, X_test, Y_train, Y_test = load_files(config)

    train_generator, mean, stddev = prepare_data(X_train, Y_train, config.batch_size, aug_data, config)

    test_generator, _1, _2 = prepare_data(X_test, Y_test, config.test_batch_size, False, config)

    valid_generator, _1, _2 = prepare_data(X_test[::2], Y_test[::2], 1, aug_data, config)

    return train_generator, test_generator, valid_generator, mean, stddev


def load_files(config = []):
    images = [f.split("/")[-1] for f in glob.glob(os.path.join("/hdd/datasets/person-seg/labels", "*.png"))]

    X = [os.path.join("/hdd/datasets/person-seg/images", f) for f in images]
    Y = [os.path.join("/hdd/datasets/person-seg/labels", f) for f in images]

    X, Y = shuffle(X, Y)

    # Use 20% of the dataset for testing, 80% for training 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

    return X_train, X_test, Y_train, Y_test

def rand_crop(img, label):
    assert img.shape[0:2] == label.shape[0:2]
    h, w = img.shape[0:2]

    if w > h:
        scale = random.randint(int(h * 0.6), h)
        top = random.randint(0, h - scale - 1)
        bottom = top + scale
        left = random.randint(0, w - scale - 1)
        right = left + scale

        return cv2.resize(img[top:bottom, left:right], (INPUTSHAPE, INPUTSHAPE)), cv2.resize(label[top:bottom, left:right], (INPUTSHAPE, INPUTSHAPE))
    else:
        scale = random.randint(int(w * 0.6), w)
        top = random.randint(0, h - scale - 1)
        bottom = top + scale
        left = random.randint(0, w - scale - 1)
        right = left + scale

        return cv2.resize(img[top:bottom, left:right], (INPUTSHAPE, INPUTSHAPE)), cv2.resize(label[top:bottom, left:right], (INPUTSHAPE, INPUTSHAPE))


def prepare_data(X, Y, batch_size, augment_data, config = []):

    X = [mpimg.imread(x) for x in X]
    X = [preprocess_image(cv2.resize(x[:, :, 0:3], (INPUTSHAPE, INPUTSHAPE)), config = config) for x in X]

    Y = [mpimg.imread(y) for y in Y]
    Y = [preprocess_label(cv2.resize(y[:], (INPUTSHAPE, INPUTSHAPE)), config = config) for y in Y]

    print("Read all data")

    def gen():
        # Generate infinite amount of data
        while True:
            i = 0

            images = np.empty([batch_size, INPUTSHAPE, INPUTSHAPE, 3])
            labels = np.empty([batch_size, INPUTSHAPE, INPUTSHAPE, 2])

            # Pick random portion of data 
            for index in np.concatenate((np.random.permutation(len(X)), np.random.permutation(len(X)))):

                image = X[index]
                label = Y[index]
                image, label = postprocess(image, label, config)

                if augment_data:
                    image, label = augment(image, label)

                image, label = rand_crop(image, label)

                newLabel = np.zeros([label.shape[0], label.shape[1], 2], dtype=np.float32)
                newLabel[np.where((label < 0.5).all(axis=2))] = (1, 0)
                newLabel[np.where((label > 0.5).all(axis=2))] = (0, 1)

                images[i] = 2 * (image - 0.5)
                labels[i] = newLabel

                # Limit number of images to batch_size
                i += 1
                if i == batch_size:
                    break


            yield images, labels

    return gen(), mean, stddev

def postprocess(image, label, config = []):
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    newLabel = np.zeros([label.shape[0], label.shape[1], 1], dtype=np.float32)

    newLabel[np.where((label < 0.5))] = 0.0
    newLabel[np.where((label > 0.5))] = 1.0

    newImage = np.array(image, dtype=np.float32)

    return newImage, newLabel
