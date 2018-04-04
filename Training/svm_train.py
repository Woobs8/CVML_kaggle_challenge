#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Tools.DataLoader import load_vector_data
from Models.SVM import SupportVectorMachine
from sklearn.metrics import accuracy_score
import argparse

training_data_path = 'Data/Train/trainVectors.txt'
training_lbls_path = 'Data/Train/trainLbls.txt'
validation_data_path = 'Data/Validation/valVectors.txt'
validation_lbls_path = 'Data/Validation/valLbls.txt'


def main(output_dir):
    # train model
    train_data, train_labels = load_vector_data(training_data_path, training_lbls_path)
    svm = SupportVectorMachine(kernel='linear', max_iter=100)
    svm.fit(train_data, train_labels, output_dir+'/fit_model.plk')
    train_pred = svm.predict(train_data, output_file=output_dir+'/train_pred.txt')
    train_acc = accuracy_score(train_labels, train_pred)

    # test model on validation set
    val_data, val_labels = load_vector_data(validation_data_path, validation_lbls_path)
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

    args = parser.parse_args()
    main(args.output)

    exit()