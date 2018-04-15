#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
from Tools.DataReader import load_vector_data
from SVM import SupportVectorMachine
from sklearn.metrics import accuracy_score
import argparse
from sklearn.preprocessing import normalize
import numpy as np


def main(output_dir, train_data_path, train_lbl_path, val_data_path, val_lbl_path, full_set):
    # data
    train_data, train_labels = load_vector_data(train_data_path, train_lbl_path)
    val_data, val_labels = load_vector_data(val_data_path, val_lbl_path)

    # normalize data
    train_data = normalize(train_data)
    val_data = normalize(val_data)

    if full_set:
        train_data = np.concatenate((train_data,val_data))
        train_labels = np.concatenate((train_labels,val_labels))

    # train model
    svm = SupportVectorMachine(kernel='rbf', C=100, gamma=0.1, max_iter=100)
    svm.fit(train_data, train_labels, output_dir+'/fit_model.plk')
    train_pred = svm.predict(train_data, output_file=output_dir+'/train_pred.txt')
    train_acc = accuracy_score(train_labels, train_pred)

    # test model on validation set
    val_pred = svm.predict(val_data, output_file=output_dir+'/val_pred.txt')
    val_acc = accuracy_score(val_labels, val_pred)

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('Classifier: SVM\n')
        fp.write('Kernel: '+ str(svm.kernel)+'\n')
        fp.write('Max iterations: '+ str(svm.max_iter)+'\n')
        fp.write('Training accuracy: ' + str(train_acc)+'\n')
        fp.write('Validation accuracy: ' + str(val_acc)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train SVM',
                                    description='''Train a SVM and store the output to the specified directory''')
    parser.add_argument('output', 
                        help='output directory where results are stored')

    parser.add_argument('train_data', 
                        help='path to training data vector', nargs='?', default='Data/Train/trainVectors.txt')

    parser.add_argument('train_label', 
                        help='path to training label vector', nargs='?', default='Data/Train/trainLbls.txt')

    parser.add_argument('val_data', 
                        help='path to validation data vector', nargs='?', default='Data/Validation/valVectors.txt')

    parser.add_argument('val_label', 
                        help='path to validation label vector', nargs='?', default='Data/Validation/valLbls.txt')

    parser.add_argument('-full_set', action='store_true')

    args = parser.parse_args()
    main(args.output, args.train_data, args.train_label, args.val_data, args.val_label, args.full_set)

    exit()