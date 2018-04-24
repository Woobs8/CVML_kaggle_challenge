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
from Tools.DataGenerator import DataGenerator
from Tools.DataReader import load_labels
from Tools.ImageReader import image_reader
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from Tools.tensorboard_lr import LRTensorBoard
from keras import backend as K
from keras.layers import Lambda, Input
from keras.callbacks import History 
K.set_image_data_format('channels_last')

def train_classifier(train_data, train_lbl, val_data, val_lbl, output_dir, tb_path, max_epochs, lr, batch_size, start_layer, stop_layer, lr_plateau =[5,0.01,1,0.000001], early_stop=[3,0.01], clf_dropout=0.5, input_model=None, print_model_summary_only=False, use_resize=False, restart=False,histogram_graphs=False):
    # load labels
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

    # no input model specified - generate new model
    if input_model is None:
        # If we dont use resize we might want to investigate using SSP to flatten the output of the last conv layer        
        # add resize layer to fit images for VGG19 input layer (224x224)
        if use_resize:
            inp = Input(shape=(None, None, 3),name='image_input')
            inp_resize = Lambda(lambda image: K.tf.image.resize_images(image, (224, 224), K.tf.image.ResizeMethod.BICUBIC),name='image_resize')(inp)
            resize = Model(inp,inp_resize)
            
            # get the VGG19 Model
            model = VGG19(input_tensor=resize.output, weights = "imagenet", include_top=True)

            """ # Dropout is data enrichment (equivalent to 2-5 times the dataset based on Alexandros)
                # Dropout should only be used during training (NOT validation and testing -> input to ouput should be deterministic)
                # If dropout is used during training the output values of the dropout layer should be multiplied with 1/(dropout_%)
                # This is due dropout not being used in validation and testing but we want the same energy to come into the FC layer after dropout
                # This should be exactly how keras implemented it from the start! https://github.com/keras-team/keras/issues/3305"""
            
            # Get the fully connected layers
            fc1 = model.get_layer('fc1')
            fc2 = model.get_layer('fc2')

            # Create the dropout layer and the classifier 
            dropout = layers.Dropout(clf_dropout, name='dropout1')  
            clf = layers.Dense(num_classes, activation="softmax",name="clf_softmax")

            # Reconnect the layers
            x = dropout(fc1.output)
            x = fc2(x)
            predictions = clf(x)

            # Create the final model
            final_model = Model(inputs = model.input, outputs = predictions)

        # use original image sizes - redefine model classification layers
        else:
            # get The VGG19 Model
            model = VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))
            # create The Classifier     
            clf = layers.Flatten()(model.output)
            clf = layers.Dense(4096, activation="relu",name="fc1")(clf)
            clf = layers.Dropout(clf_dropout, name="dropout1")(clf)
            clf = layers.Dense(4096, activation="relu", name="fc2")(clf)
            predictions = layers.Dense(num_classes, activation="softmax",name="clf_softmax")(clf)
            
            final_model = Model(input = model.input, output = predictions)
    # load input model
    else:
        print("Using existing model: {}".format(input_model))
        final_model = load_model(input_model)

    if not restart:    
        # freeze all layers, so the trainable layers are controlled
        for layer in final_model.layers:
            layer.trainable = False
        

    # Data generators
    train_generator = DataGenerator(path_to_images=train_data,
                                    labels=cat_train_labels, 
                                    batch_size=batch_size)
        
    if histogram_graphs: 
        # If we want histogram graphs we must pass all val images as numpy array
        hist_frq = 1
        validation_images = image_reader(val_data)*(1./255)
        val_generator = (validation_images, cat_val_labels)
        val_steps = None
    else:
        hist_frq = 0
        val_steps = len(validation_labels)/batch_size
        val_generator = DataGenerator(  path_to_images=val_data,
                                        labels=cat_val_labels, 
                                        batch_size=batch_size)
    
    # define model keras callbacks 
    checkpoint = ModelCheckpoint(filepath=output_dir+"/checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=early_stop[1], patience=early_stop[0], verbose=1, mode='auto')
    tensorboard = LRTensorBoard(log_dir=tb_path, histogram_freq=hist_frq, write_graph=True, write_grads=True,batch_size=batch_size, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    plateau = ReduceLROnPlateau(monitor='val_loss', factor=lr_plateau[2], patience=lr_plateau[0], verbose=0, mode='auto', epsilon=lr_plateau[1], cooldown=1, min_lr=lr_plateau[3])
    history = History()
    
    # Create List of Callbacks
    callback_list = [checkpoint, early, tensorboard, plateau, history]
    
    if not restart:
        # set trainable layers
        start_layer_idx = [i for i,j in enumerate(final_model.layers) if j.name==start_layer][0]
        stop_layer_idx = [i for i,j in enumerate(final_model.layers) if j.name==stop_layer][0]
        
        for layer in final_model.layers[stop_layer_idx:start_layer_idx+1]:
            layer.trainable = True
                
        # compile the model 
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True), metrics=["accuracy"])

    # print model summary and stop if specified
    final_model.summary()
    if print_model_summary_only:
        return 

    # fit model
    final_model.fit_generator(train_generator,
                        steps_per_epoch = len(training_labels)/batch_size,
                        epochs = max_epochs,
                        validation_data = val_generator,
                        validation_steps = val_steps,
                        callbacks = callback_list,
                        workers=1, # Only use one worker or the batches will be dublicates of each other 
                        use_multiprocessing=True)
    
    print("Finished training layers: {} - {}".format(start_layer,stop_layer), flush=True)

    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('Max epochs: '+ str(max_epochs)+'\n')
        fp.write('lr: '+ str(tensorboard.get_lr_hist())+'\n')
        fp.write('Batch size: '+ str(batch_size)+'\n')
        fp.write('Starting layer: ' + str(start_layer)+'\n')
        fp.write('Stopping layer: ' + str(stop_layer)+'\n')
        fp.write('Training accuracy: ' + str(history.history['acc'])+'\n')
        fp.write('Validation accuracy: ' + str(history.history['val_acc'])+'\n')
    
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train specific layers of a VGG19 keras model',
                                    description='''Train specific layers of a VGG19 keras model. The model may be trained from scratch, pretrained or an existing model saved in an external file''')

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

    parser.add_argument('-tb_path', 
                        help='output directory where tensorboard results are stored. The folder structure should be /logdir/Run1/ , /logdir/Run2/ etc. ',
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
   
    parser.add_argument('-lr_plateau',
                        help='Parameters for applying learn rate drop on \'val loss\' plateau schedule (patience, min_delta, drop factor, min_lr)',
                        nargs=4,
                        default=[5,0.01,0.1,0.000001],
                        type=float,
                        required=False) 

    parser.add_argument('-early_stop',
                        help='Parameters for early stopping (patience, decimal change in val_acc)',
                        nargs=2,
                        type=float,
                        default=[3,0.01],
                        required=False) 

    parser.add_argument('-dropout', 
                        help='Dropout rate to use',
                        type=float,
                        default=0.5)

    parser.add_argument('-batch_size', 
                        help='Batch size to use when training',
                        type=int,
                        default=32)
                        
    parser.add_argument('-input_model', 
                        help='Path to .h5 model to train last layers',
                        default=None)
    
    parser.add_argument('-restart', 
                        help='Make sure that the model use loaded learn rate and architecture (model wont be compiled)',
                        type=str2bool,
                        default=False)

    parser.add_argument('-use_resize', 
                        help='Use resizing to 224*224*3',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('-histogram_graphs', 
                        help='Dropout rate to use',
                        type=str2bool,
                        default=False)   

    parser.add_argument('-summary_only', 
                        help='Stop Script after prining model summary (ie. no training)',
                        type=str2bool,
                        default=False)

    args = parser.parse_args()
    
    train_classifier(   train_data=args.train_data, 
                        train_lbl=args.train_label, 
                        val_data=args.val_data, 
                        val_lbl=args.val_label, 
                        output_dir=args.output,
                        max_epochs=args.epochs, 
                        lr=args.lr, 
                        batch_size=args.batch_size, 
                        lr_plateau=args.lr_plateau,
                        input_model=args.input_model,
                        start_layer=args.start_layer,
                        stop_layer=args.stop_layer,
                        clf_dropout=args.dropout,
                        print_model_summary_only=args.summary_only,
                        use_resize=args.use_resize,
                        restart=args.restart,
                        early_stop=args.early_stop,
                        tb_path=args.tb_path,
                        histogram_graphs=args.histogram_graphs)