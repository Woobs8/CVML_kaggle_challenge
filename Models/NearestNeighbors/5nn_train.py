#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from Tools.DataReader import load_vector_data
from KNN import KNearestNeighbors
from sklearn.metrics import accuracy_score
import argparse
import numpy as np


def main(output_dir, train_data_path, train_lbl_path, val_data_path, val_lbl_path):
    # train model
    train_data, train_labels = load_vector_data(train_data_path, train_lbl_path)
    knn = KNearestNeighbors(5)
    knn.fit(train_data, train_labels, output_dir+'/fit_model.plk')
    train_pred = knn.predict(train_data, output_file=output_dir+'/train_pred.txt')
    train_acc = accuracy_score(train_labels, train_pred)

    # test model on validation set
    val_data, val_labels = load_vector_data(val_data_path, val_lbl_path)
    val_pred = knn.predict(val_data, output_file=output_dir+'/val_pred.txt')
    val_acc = accuracy_score(val_labels, val_pred)

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('Classifier: KNN\n')
        fp.write('Neighbors: '+ str(knn.neighbors)+'\n')
        fp.write('Neighbor weights : '+ str(knn.neighbor_weights)+'\n')
        fp.write('Training accuracy: ' + str(train_acc)+'\n')
        fp.write('Validation accuracy: ' + str(val_acc)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train KNN classifier',
                                    description='''Train a KNN classifier and store the output to the specified directory''')
    parser.add_argument('output', 
                        help='output directory where results are stored')

    parser.add_argument('train_data', 
                        help='path to training data vector', nargs='?', default='../../Data/Train/trainVectors.txt')

    parser.add_argument('train_label', 
                        help='path to training label vector', nargs='?', default='../../Data/Train/trainLbls.txt')

    parser.add_argument('val_data', 
                        help='path to validation data vector', nargs='?', default='../../Data/Validation/valVectors.txt')

    parser.add_argument('val_label', 
                        help='path to validation label vector', nargs='?', default='../../Data/Validation/valLbls.txt')

    args = parser.parse_args()
    main(args.output, args.train_data, args.train_label, args.val_data, args.val_label)

    exit()