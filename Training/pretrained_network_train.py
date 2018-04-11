#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Tools.ImageReader import image_reader
from Tools.LearningRate import step_decay
from Models.CNN import PretrainedConvolutionalNeuralNetwork, SUPPORTED_ARCHITECTURES
from sklearn.metrics import accuracy_score
import argparse
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras import optimizers

training_data_path = 'Data/Train/TrainImages'
validation_data_path = 'Data/Validation/ValidationImages'

training_lbls_path = 'Data/Train/trainLbls.txt'
validation_lbls_path = 'Data/Validation/valLbls.txt'

def train_pretrained_network(output_dir, data_augmentation, nb_epochs, init_learn_rate, learn_rate_drop, epochs_btw_drop, pretrained_architecture,training_batch_size, dropout, optimizer):
    # Load images
    training_data, training_labels = image_reader(training_data_path,training_lbls_path)
    validation_data, validation_labels = image_reader(validation_data_path,validation_lbls_path)
    
    # Define LearningRateScheduler
    lrate = LearningRateScheduler(step_decay(nb_epochs, init_learn_rate, learn_rate_drop, epochs_btw_drop))

    # Create Optimizer
    if optimizer == 'SGD':
        opti = optimizers.SGD(lr=init_learn_rate, momentum=0.9, nesterov=True)
    elif optimizer == 'RMSprop':
        opti = optimizers.RMSprop(lr=init_learn_rate, rho=0.9, epsilon=None)
    elif optimizer == 'Adam':
        opti = optimizers.Adam(lr=init_learn_rate, beta_1=0.9, beta_2=0.999)   

    # train model
    clf = PretrainedConvolutionalNeuralNetwork(optimizer=opti,epochs=nb_epochs, batch_size=training_batch_size, dropout=dropout, architecture=pretrained_architecture, data_augmentation=data_augmentation, img_width = 256, img_height = 256,img_depth=3)
    hist = clf.fit(training_data,training_labels,val_data=validation_data, val_labels=validation_labels, lr_schedule=lrate, log_dir=output_dir)
    
    train_acc = hist['acc'][-1]
    val_acc = hist['val_acc'][-1]

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write(pretrained_architecture+' Neaural Network Training Parameters:\n')
        fp.write('Learning rate schedule: step decay')
        fp.write('Initial Learning rate: '+ str(init_learn_rate)+'\n')
        fp.write('Decay: '+ str(learn_rate_drop)+'\n')
        fp.write('Epoch drop: '+ str(epochs_btw_drop)+'\n')
        fp.write('Optimizer: '+ optimizer +'\n')
        fp.write('Batch size: '+ str(clf.batch_size)+'\n')
        fp.write('Epochs: '+ str(clf.epochs)+'\n')
        fp.write('Training accuracy: ' + str(train_acc)+'\n')
        fp.write('Validation accuracy: ' + str(val_acc)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train on pretrained convolutional neural network',
                                    description='''Train last layers of pretrained convolutional neural network and store the output to the specified directory''')
    
    parser.add_argument('output', 
                        help='output directory where results are stored')
    
    parser.add_argument('-data_augmentation', 
                        help='output directory where results are stored',
                        action='store_true')
    
    parser.add_argument('-nb_epochs', 
                        help='Max number of epochs to run',
                        default=20)

    parser.add_argument('-init_learn_rate', 
                        help='Initial learning rate',
                        default=0.01)
    
    parser.add_argument('-learn_rate_drop', 
                        help='Factor that the learn rate drops',
                        default=0.5)
    
    parser.add_argument('-epochs_btw_drop', 
                        help='Number of epochs between learn rate drop',
                        default=10)
    
    parser.add_argument('-pretrained_architecture', 
                        help='Choose pretrained architecture',
                        choices = SUPPORTED_ARCHITECTURES,
                        default = SUPPORTED_ARCHITECTURES[0])
    
    parser.add_argument('-training_batch_size', 
                        help='Batch size to use when training',
                        default=32)

    parser.add_argument('-dropout', 
                        help='Dropout percentage',
                        default=0.5)
    
    parser.add_argument('-optimizer', 
                        help='Choose optimization algorithm',
                        choices=['SGD','RMSprop','Adam'],
                        default='SGD')

    args = parser.parse_args()
    
    train_pretrained_network(args.output,
                            data_augmentation=args.data_augmentation,
                            nb_epochs=args.nb_epochs,
                            init_learn_rate=args.init_learn_rate,
                            learn_rate_drop=args.learn_rate_drop,
                            epochs_btw_drop=args.epochs_btw_drop,
                            pretrained_architecture=args.pretrained_architecture,
                            training_batch_size=args.training_batch_size,
                            dropout=args.dropout,
                            optimizer=args.optimizer)
    exit()