#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
from Tools.DataReader import load_data
from Tools.DataWriter import write_predictions_file
from SVM import SupportVectorMachine
from sklearn.metrics import accuracy_score
import argparse
from sklearn.preprocessing import normalize
import numpy as np


def main(model_path, output_file_path, test_data_path, norm):
    # data
    test_data = load_data(test_data_path)

    # normalize data
    if norm:
        test_data = normalize(test_data)

    # load model
    svm = SupportVectorMachine()
    svm.load(model_path)
    test_pred = svm.predict(test_data, instance=True, decision='average')

    write_predictions_file(test_pred, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Predict samples using trained SVM model',
                                    description='''Predict samples using a trained SVM and store the predictions in the the specified file''')
    
    parser.add_argument('model', 
                        help='path to model')

    parser.add_argument('output', 
                        help='output directory where results are stored')

    parser.add_argument('test_data', 
                        help='path to test data vector', nargs='?', default='Data/Test/testVectors.txt')

    parser.add_argument('-norm', action='store_true')

    args = parser.parse_args()
    main(args.model, args.output, args.test_data, args.norm)

    exit()