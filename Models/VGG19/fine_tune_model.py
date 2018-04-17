#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras.layers import Input, Lambda
from keras import optimizers,layers
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.applications import VGG19
from Tools.LearningRate import step_decay
from Tools.DataGenerator import DataGenerator
from Tools.DataReader import load_labels
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K


def fine_tune_model(train_data, train_lbl, val_data, val_lbl, model_path, output_dir, retrain_layer_name, max_epochs, init_lr, batch_size, lr_sched=None, print_model_summary_only=False,compile_model=False):
    # Load labels
    training_labels = load_labels(train_lbl)
    validation_labels = load_labels(val_lbl)
    
    # labels must be from 0-num_classes-1, so label offset is subtracted
    unique, count = np.unique(training_labels,return_counts=True) 
    num_classes = len(unique)
    label_offset = int(unique[0])
    training_labels -= label_offset
    validation_labels -= label_offset

    # one-hot encode labels
    cat_train_labels = to_categorical(training_labels)
    cat_val_labels = to_categorical(validation_labels)

    # Create Layers in case they are part of the model
    inp = Input(shape=(None, None, 3),name='image_input')
    inp_resize = Lambda(lambda image: ktf.image.resize_images(image, (224, 224), ktf.image.ResizeMethod.BICUBIC),name='image_resize')(inp)
    resize = Model(inp,inp_resize)

    # load pre-trained model
    print("load model")
    print(model_path)
    final_model = load_model(model_path)
    print("model loaded")
    # freeze the specified layers of the pre-trained model
    retrain_flag = False
    if retrain_layer_name.isdigit():
        for layer in final_model.layers[:-int(retrain_layer_name)]:
            layer.trainable = False
        for layer in final_model.layers[-int(retrain_layer_name):]:
            layer.trainable = True
    elif retrain_layer_name.lower() == 'all':
        for layer in final_model.layers:
            layer.trainable = False
    else:
        for layer in final_model.layers:
            if layer.name == retrain_layer_name:
                retrain_flag = True
            if retrain_flag:
                layer.trainable = True
            else:
                layer.trainable = False

    if compile_model:
        # If the model is compiled the Optimizer states are overwritten (does not start from where it ended)
        print("compile")
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=init_lr,momentum=0.9,nesterov=True), metrics=["accuracy"])
    
    final_model.summary()
    
    if print_model_summary_only:
        return
    
    # define model callbacks 
    checkpoint = ModelCheckpoint(filepath=output_dir+"/checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=1, mode='auto')
    tb_path = os.path.join(output_dir,'Graph')
    tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
    
    if not lr_sched is None:
        lrate = LearningRateScheduler(step_decay(init_lr, lr_sched[0], lr_sched[1]))
        callback_list = [checkpoint, early, tensorboard, lrate]
    else:
        callback_list = [checkpoint, early, tensorboard]

    # Data generators
    train_generator = DataGenerator(path_to_images=train_data,
                                    labels=cat_train_labels, 
                                    batch_size=batch_size)
    val_generator = DataGenerator(  path_to_images=val_data,
                                    labels=cat_val_labels, 
                                    batch_size=batch_size)

    # fit model
    final_model.fit_generator(train_generator,
                        steps_per_epoch = len(training_labels)/batch_size,
                        epochs = max_epochs,
                        validation_data = val_generator,
                        validation_steps = len(validation_labels)/batch_size,
                        callbacks = callback_list,
                        workers=2,
                        use_multiprocessing=False)
    
    print("Finished Training")

    # save final model
    final_model.save(output_dir+"/final_model.h5")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Fine-tune pre-trained network and existing classifier',
                                    description='''Train last layers of pretrained convolutional neural network and store the output to the specified directory''')

    parser.add_argument('train_data', 
                        help='path to directory containing training images', 
                        nargs='?', 
                        default='Data/Train/TrainImages')

    parser.add_argument('train_label', 
                        help='path to training label vector', 
                        nargs='?', 
                        default='Data/Train/trainLbls.txt')

    parser.add_argument('val_data', 
                        help='path to directory containing validation images', 
                        nargs='?', 
                        default='Data/Validation/ValidationImages')

    parser.add_argument('val_label', 
                        help='path to validation label vector', 
                        nargs='?', 
                        default='Data/Validation/valLbls.txt')

    parser.add_argument('-output', 
                        help='output directory where results are stored',
                        required=True)

    parser.add_argument('-input_model', 
                        help='Path to .h5 model to train last layers. First layer of the top layers to train must be named clf_dense_1',
                        required=True)
    
    parser.add_argument('-retrain_layer_name', 
                        help="The layer name from where the model should be fine-tuned, \"all\" is a valid value, where all layers will be frozen including the classifier. If a number is given the last \"x\" number of the network will be retrained.",
                        default='All')

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

    parser.add_argument('-compile', 
                        help='Stop Script after prining model summary (ie. no training)',
                        action="store_true")

    parser.add_argument('-summary_only', 
                        help='Stop Script after prining model summary (ie. no training)',
                        action="store_true")

    args = parser.parse_args()
    
    fine_tune_model(train_data=args.train_data, 
                    train_lbl=args.train_label, 
                    val_data=args.val_data, 
                    val_lbl=args.val_label, 
                    model_path=args.input_model,
                    output_dir=args.output,
                    retrain_layer_name=args.retrain_layer_name,
                    max_epochs=args.epochs, 
                    init_lr=args.init_lr, 
                    batch_size=args.batch_size, 
                    lr_sched=args.lr_sched,
                    print_model_summary_only=args.summary_only,
                    compile_model=args.compile)