#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
from keras.models import load_model
from keras import optimizers
from keras.callbacks import LearningRateScheduler


def fine_tune_model(train_data, train_lbl, val_data, val_lbl, clf_path, model_path, output_dir, freeze_layers, max_epochs, init_lr, batch_size, lr_sched=None):
    # Load data
    training_data, training_labels = image_reader(training_data_path,training_lbls_path)
    validation_data, validation_labels = image_reader(validation_data_path,validation_lbls_path)
    
    # labels must be from 0-num_classes-1, so label offset is subtracted
    label_offset = int(unique[0])
    train_labels -= label_offset
    val_labels -= label_offset

    # one-hot encode labels
    cat_train_labels = to_categorical(train_labels)
    cat_val_labels = to_categorical(val_labels)

    # load pre-trained model
    model = load_model(model_path)

    # freeze the specified layers of the pre-trained model
    for layer in model.layers[:freeze_layers]
        layer.trainable = False

    # load classifier and add it on top of pre-trained model
    clf = load_model(clf_path)
    model.add(clf)

    # compile the model 
    model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=init_lr, momentum=0.9, nesterov=True), metrics=["accuracy"])

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

    # TODO: Data generators

    # fit model
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_data) / batch_size,
                        epochs = max_epochs,
                        validation_data = validation_generator,
                        validation_steps = len(val_data)/batch_size,
                        callbacks = callback_list)

    # save final model
    model.save_weights('final_model.h5')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Fine-tune pre-trained network and existing classifier',
                                    description='''Train last layers of pretrained convolutional neural network and store the output to the specified directory''')

    parser.add_argument('train_data', 
                        help='path to training data vector', nargs='?', default='../../Data/Train/trainVectors.txt')

    parser.add_argument('train_label', 
                        help='path to training label vector', nargs='?', default='../../Data/Train/trainLbls.txt')

    parser.add_argument('val_data', 
                        help='path to validation data vector', nargs='?', default='../../Data/Validation/valVectors.txt')

    parser.add_argument('val_label', 
                        help='path to validation label vector', nargs='?', default='../../Data/Validation/valLbls.txt')

    parser.add_argument('classifier', 
                        help='path to rough-tuned classifier')

    parser.add_argument('model', 
                        help='path to pretrained model')

    parser.add_argument('output', 
                        help='output directory where results are stored')
    
    parser.add_argument('-freeze_layers', 
                        help='Number of layers in pre-trained model to freeze during fine-tuning',
                        type=int,
                        default='SGD')

    parser.add_argument('-epochs', 
                        help='Max number of epochs to run',
                        type=int,
                        default=20)

    parser.add_argument('-init_lr', 
                        help='Initial learning rate',
                        type=float,
                        default=0.01)
    
    parser.add_argument('-lr_sched',
                        help='Parameters for learning rate schedule (drop, epochs between drop)',
                        nargs=2,
                        type=float,
                        required=False)        
    
    parser.add_argument('-batch_size', 
                        help='Batch size to use when training',
                        type=int,
                        default=32)

    args = parser.parse_args()
    
    fine_tune_model(train_data=args.train_data, 
                    train_lbl=args.train_label, 
                    val_data=args.val_data, 
                    val_lbl=args.val_label, 
                    clf_path=args.classifier, 
                    model_path=args.model,
                    output_dir=args.output,
                    freeze_layers=args.freeze_layers,
                    max_epochs=args.epochs, 
                    init_lr=args.init_lr, 
                    batch_size=args.batch_size, 
                    lr_sched=args.lr_sched)