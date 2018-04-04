#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

def load_vector_data(data_path, label_path):
    # load data
    path, file_extension = os.path.splitext(data_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        data = np.load(path_numpy).transpose()
    else:
        data = np.loadtxt(path+file_extension,delimiter=' ')
        np.save(path_numpy,data)

    # load labels
    path, file_extension = os.path.splitext(label_path)
    path_numpy = path+'.npy'
    if os.path.exists(path_numpy):
        labels = np.load(path_numpy)
    else:
        labels = np.loadtxt(path+file_extension,delimiter=' ')
        np.save(path_numpy,labels)
    return data, labels