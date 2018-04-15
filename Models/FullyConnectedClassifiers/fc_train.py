#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
from Tools.DataReader import load_vector_data
from Tools.LearningRate import step_decay
from Tools.DataGenerator import DataGenerator
import argparse
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import normalize      


def train_fc_classifier(train_data_path, train_lbl_path, val_data_path, val_lbl_path, layers, dim, output_dir, max_epochs, init_lr, dropout, batch_size, lr_sched=None):
    # load data
    train_data, train_labels = load_vector_data(train_data_path, train_lbl_path)
    val_data, val_labels = load_vector_data(val_data_path,val_lbl_path)



    # labels must be from 0-num_classes-1, so label offset is subtracted
    unique, count = np.unique(train_labels,return_counts=True) 
    num_classes = len(unique)
    label_offset = int(unique[0])
    train_labels -= label_offset
    val_labels -= label_offset
    
    # determine class weights to account for difference in samples for classes
    num_samples, num_features = train_data.shape
    class_weights = num_samples/count
    normalized_class_weights = class_weights / np.max(class_weights)
    class_weights = dict(zip(unique-label_offset, normalized_class_weights))

    # generate list of layer dimensions
    dim_list = [dim]*layers

    model = Sequential()
    prev_output_dim = num_features
    for idx, layer in enumerate(range(1,layers+1)):
        model.add(Dense(dim_list[idx], input_dim=prev_output_dim, activation='relu'))
        # add dropout between hidden layers
        if layer < layers:
            model.add(Dropout(dropout))
        prev_output_dim = dim_list[idx]
    model.add(Dense(num_classes, input_dim=prev_output_dim, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', 
                        optimizer=optimizers.SGD(lr=init_lr, momentum=0.9, nesterov=True), 
                        metrics=['accuracy'])

    # print summary of model architecture
    print(model.summary())

    # one-hot encode labels
    cat_train_labels = to_categorical(train_labels)
    cat_val_labels = to_categorical(val_labels)

    # define model callbacks 
    checkpoint = ModelCheckpoint("checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    tb_path = os.path.join(output_dir,'Graph')
    tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)

    if not lr_sched is None:
        lrate = LearningRateScheduler(step_decay(init_lr, lr_sched[0], lr_sched[1]))
        callback_list = [checkpoint, early, tensorboard, lrate]
    else:
        callback_list = [checkpoint, early, tensorboard]
                                    
    # fit model
    model.fit(train_data,
            cat_train_labels,  
            epochs = max_epochs,
            validation_data = (val_data,cat_val_labels),
            callbacks = callback_list,
            class_weight=class_weights, 
            batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train fully-connected neural network',
                                    description='''Train a fully-connected neural network classifier and store the output in the specified directory''')
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

    parser.add_argument('-layers', 
                        help='Number of fully-connected layers (excluding softmax classifier)',
                        type=int,
                        default=0)                    

    parser.add_argument('-dim', 
                        help='Dimensions of fully-connected layers',
                        type=int,
                        default=512)                    
    
    parser.add_argument('-epochs', 
                        help='Max number of epochs to run',
                        type=int,
                        default=20)

    parser.add_argument('-init_lr', 
                        help='Initial learning rate',
                        type=float,
                        default=0.01)

    parser.add_argument('-dropout', 
                        help='Specify classifier dropout',
                        type=float,
                        default=0.5)
   
    parser.add_argument('-lr_sched',
                        help='Parameters for learning rate schedule (drop, epochs between drop)',
                        nargs=2,
                        type=float,
                        required=False)        
    
    parser.add_argument('-batch_size', 
                        help='Batch size to use when training',
                        type=int,
                        default=32)

        
    parser.add_argument('-norm', 
                        help='Normalize data before training classifier',
                        action='store_true')   

    args = parser.parse_args()
    train_fc_classifier(train_data_path=args.train_data, 
                        train_lbl_path=args.train_label, 
                        val_data_path=args.val_data, 
                        val_lbl_path=args.val_label,
                        layers=args.layers,
                        dim=args.dim, 
                        output_dir=args.output,
                        max_epochs=args.epochs, 
                        init_lr=args.init_lr, 
                        dropout=args.dropout,
                        batch_size=args.batch_size, 
                        lr_sched=args.lr_sched)

    exit()