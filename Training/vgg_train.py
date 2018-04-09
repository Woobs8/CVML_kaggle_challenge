#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Tools.DataReader import load_vector_data
from Tools.LearningRate import step_decay
from Models.CNN import PretrainedConvolutionalNeuralNetwork
from sklearn.metrics import accuracy_score
import argparse
from keras.callbacks import LearningRateScheduler
import numpy as np      

training_data_path = 'Data/Train/TrainImages'
validation_data_path = 'Data/Validation/ValidationImages'


def main(output_dir):
    # define learning rate
    epochs = 1000
    init_lr = 0.01
    lr_drop = 0.5
    epochs_drop = 10
    lrate = LearningRateScheduler(step_decay(epochs,init_lr,lr_drop,epochs_drop))

    # train model
    clf = PretrainedConvolutionalNeuralNetwork(num_classes = 29,epochs=epochs, batch_size=128, dropout=0.5, architecture="VGG16", data_augmentation=False, num_freeze_layers=16, img_width = 256, img_height = 256,img_depth=3)
    hist = clf.fit(training_data_path, validation_data_path,steps_per_epoch = 4*1024, validation_steps=512, lr_schedule=lrate, log_dir=output_dir)
    
    train_acc = hist['acc'][-1]
    val_acc = hist['val_acc'][-1]

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('VGG Neaural Network Training Parameters:\n')
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
    parser = argparse.ArgumentParser(prog='Train VGG16 convolutional neural network',
                                    description='''Train VGG16 convolutional neural network and store the output to the specified directory''')
    parser.add_argument('output', 
                        help='output directory where results are stored')

    args = parser.parse_args()
    main(args.output)

    exit()