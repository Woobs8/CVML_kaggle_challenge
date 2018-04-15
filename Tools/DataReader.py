#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.utils import shuffle

def load_vector_data(data_path, label_path, shuffle_data=True):
    # load data
    path, file_extension = os.path.splitext(data_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        data = np.load(path_numpy)
    else:
        data = np.loadtxt(path+file_extension,delimiter=' ').transpose()
        np.save(path_numpy,data)

    # load labels
    path, file_extension = os.path.splitext(label_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        labels = np.load(path_numpy)
    else:
        labels = np.loadtxt(path+file_extension,delimiter=' ')
        np.save(path_numpy,labels)

    # shuffle data and labels
    if shuffle_data:
        data, labels = shuffle(data,labels)
    return data, labels

def load_labels(label_path):
    # load labels
    path, file_extension = os.path.splitext(label_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        labels = np.load(path_numpy)
    else:
        labels = np.loadtxt(path+file_extension,delimiter=' ')
        np.save(path_numpy,labels)

    return labels

def load_data(data_path):
    # load data
    path, file_extension = os.path.splitext(data_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        data = np.load(path_numpy)
    else:
        data = np.loadtxt(path+file_extension,delimiter=' ').transpose()
        np.save(path_numpy,data)
    
    return data