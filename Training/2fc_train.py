#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Tools.DataLoader import load_vector_data
from Tools.LearningRate import step_decay
from Models.FC import FullyConnectedClassifier
from sklearn.metrics import accuracy_score
import argparse
from keras.callbacks import LearningRateScheduler
import numpy as np      

training_data_path = 'Data/Train/trainVectors.txt'
training_lbls_path = 'Data/Train/trainLbls.txt'
validation_data_path = 'Data/Validation/valVectors.txt'
validation_lbls_path = 'Data/Validation/valLbls.txt'


def main(output_dir):
    # load data
    train_data, train_labels = load_vector_data(training_data_path, training_lbls_path)
    val_data, val_labels = load_vector_data(validation_data_path, validation_lbls_path)

    # define learning rate
    epochs = 1000
    init_lr = 0.01
    lr_drop = 0.5
    epochs_drop = 10
    lrate = LearningRateScheduler(step_decay(epochs,init_lr,lr_drop,epochs_drop))

    # train model
    clf = FullyConnectedClassifier(epochs=epochs, hidden_layers=2, dimensions=[4096,4096], batch_size=128, dropout=0.5)
    hist = clf.fit(train_data, train_labels, lr_schedule=lrate, val_data=val_data, val_labels=val_labels, log_dir=output_dir, class_weighting=True)
    train_acc = hist['acc'][-1]
    val_acc = hist['val_acc'][-1]

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('Classifier: Softmax\n')
        fp.write('Learning rate schedule: step decay')
        fp.write('Initial Learning rate: '+ str(init_lr)+'\n')
        fp.write('Decay: '+ str(lr_drop)+'\n')
        fp.write('Epoch drop: '+ str(epochs_drop)+'\n')
        fp.write('Momentum: '+ str(clf.momentum)+'\n')
        fp.write('Batch size: '+ str(clf.batch_size)+'\n')
        fp.write('Epochs: '+ str(clf.epochs)+'\n')
        fp.write('Training accuracy: ' + str(train_acc)+'\n')
        fp.write('Validation accuracy: ' + str(val_acc)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train Softmax classifier',
                                    description='''Train a softmax classifier and store the output to the specified directory''')
    parser.add_argument('output', 
                        help='output directory where results are stored')

    args = parser.parse_args()
    main(args.output)

    exit()