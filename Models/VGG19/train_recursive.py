#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras import optimizers,layers
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.applications import VGG19
from Tools.LearningRate import step_decay
from Tools.DataGenerator import DataGenerator
from Tools.DataReader import load_labels
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K
from keras.backend import tf as ktf
from keras.layers import Lambda, Input



def train_classifier(train_data, train_lbl, val_data, val_lbl, output_dir, max_epochs, lr, batch_size, start_layer, stop_layer, lr_sched=None, input_model=None, print_model_summary_only=False,use_resize=False):
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

    if input_model is None:
        # Get The VGG19 Model
        model = VGG19(input_tensor=resize.output, weights = "imagenet", include_top=True)

        if use_resize:
            inp = Input(shape=(None, None, 3),name='image_input')
            inp_resize = Lambda(lambda image: ktf.image.resize_images(image, (224, 224), ktf.image.ResizeMethod.BICUBIC),name='image_resize')(inp)
            resize = Model(inp,inp_resize)

            # Throw away softmax
            model.layers.pop()
            
            # Create The Classifier     
            predictions = layers.Dense(num_classes, activation="softmax",name="clf_softmax")(model.layers[-1].output)
            final_model = Model(input = model.input, output = predictions)

            # freeze all layers, only the classifier is trained 
            for layer in final_model.layers:
                layer.trainable = False
        else:
            # Create The Classifier     
            clf = layers.Flatten()(model.output)
            clf = layers.Dense(4096, activation="relu",name="clf_dense_1")(clf)
            clf = layers.Dropout(clf_dropout,name="clf_dropout")(clf)
            clf = layers.Dense(4096, activation="relu",name="clf_dense_2")(clf)
            predictions = layers.Dense(num_classes, activation="softmax",name="clf_softmax")(clf)
            
            final_model = Model(input = model.input, output = predictions)
        
        # compile the model 
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True), metrics=["accuracy"])

    else:
        final_model = load_model(input_model)
    
    # freeze all layers, so that no unexpected layers are trained
    for layer in final_model.layers:
        layer.trainable = False

    # Print model summary and stop if specified
    print(final_model.summary())
    if print_model_summary_only:
        return 

    # define model callbacks 
    checkpoint = ModelCheckpoint(filepath=output_dir+"/checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')
    tb_path = os.path.join(output_dir,'Graph')
    tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
    
    # Use Learn rate scheduler if specified
    if not lr_sched is None:
        lrate = LearningRateScheduler(step_decay(lr, lr_sched[0], lr_sched[1]))
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
    
    # Get list of layers to train recursively
    train_layer_list = []
    train_flag = False
    for layer in final_model.layers:
        if layer.name == stop_layer:
            train_flag = True
        
        if train_flag:
            if layer.count_params() > 0:
                train_layer_list.append(layer.name)
            
            if layer.name == start_layer:
                break
    
    # Reverse the list
    train_layer_list = train_layer_list[::-1]

    # Recursive training of the network.
    for idx, retrain_now in enumerate(train_layer_list):      
        for layer in final_model.layers:
            if layer.name == retrain_now:  
                layer.trainable = True
            else:
                layer.trainable = False
            
        # compile the model 
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True), metrics=["accuracy"])

        # fit model
        final_model.fit_generator(train_generator,
                            steps_per_epoch = len(training_labels)/batch_size,
                            epochs = max_epochs,
                            validation_data = val_generator,
                            validation_steps = len(validation_labels)/batch_size,
                            callbacks = callback_list,
                            workers=2,
                            use_multiprocessing=False)
        
        print("Finished training for layer: %s" %(retrain_now), flush=True)


    # save final model
    final_model.save(output_dir + "/recursive_clf_vgg19.h5")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Fine-tune pre-trained network recursively',
                                    description='''Recursively train specified layers of pretrained convolutional neural network and store the output to the specified directory''')

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

    parser.add_argument('-stop_layer', 
                        help='Layer to start recursive training from (layer is included in training)',
                        required=True)

    parser.add_argument('-start_layer', 
                        help='Layer to stop recursive training from (layer is included in training)',
                        required=True)  

    parser.add_argument('-epochs', 
                        help='Max number of epochs to run',
                        type=int,
                        default=20)

    parser.add_argument('-lr', 
                        help='Learning rates for each training iteration',
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
                        
    parser.add_argument('-input_model', 
                        help='Path to .h5 model to train last layers. First layer of the top layers to train must be named clf_dense_1',
                        default=None)

    parser.add_argument('-summary_only', 
                        help='Stop Script after prining model summary (ie. no training)',
                        action="store_true")
    
    parser.add_argument('-use_resize', 
                        help='Use resizing to 224*224*3',
                        action="store_true")

    args = parser.parse_args()
    
    train_classifier(   train_data=args.train_data, 
                        train_lbl=args.train_label, 
                        val_data=args.val_data, 
                        val_lbl=args.val_label, 
                        output_dir=args.output,
                        max_epochs=args.epochs, 
                        lr=args.lr, 
                        batch_size=args.batch_size, 
                        lr_sched=args.lr_sched,
                        input_model=args.input_model,
                        start_layer=args.start_layer,
                        stop_layer=args.stop_layer,
                        print_model_summary_only=args.summary_only,
                        use_resize=args.use_resize)