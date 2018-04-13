import numpy as np
import os, argparse, sys, re, random
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
from shutil import move
from scipy.misc import imread
from Tools.DataReader import load_labels
from Tools.ImageReader import image_reader, create_image_lists
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def data_generator(path_to_images, labels, batch_size):
    # Get File names and labels locally
    imglist = create_image_lists(path_to_images)
    # Random shuffle
    permutation = np.random.permutation(len(labels))
    imglist = [imglist[i] for i in permutation]
    labels = labels[permutation]

    while True:
        for X_batch, y_batch in get_batches(imglist, labels, batch_size):
            yield X_batch, y_batch

def get_batches(imglist, labels, batch_size):
    # Get Next Batch
    X = np.array([np.array(imread(fname)) for fname in imglist[:batch_size]])
    y = labels[:len(X)]
    # Place most recent batch of samples in the end of the array
    imglist[:] = imglist[batch_size:] + imglist[:batch_size]
    labels[:] = np.concatenate([labels[batch_size:], labels[:batch_size]])
    print("here")
    print(X.shape)
    print(y.shape)
    yield X, y
    


    


