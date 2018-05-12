#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras import optimizers,layers,regularizers
from keras.models import Model, load_model, clone_model
from keras.utils import to_categorical
from keras.applications import InceptionResNetV2
from Tools.DataGenerator import DataGenerator
from Tools.DataReader import load_labels
from Tools.ImageReader import image_reader
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from Tools.tensorboard_lr import LRTensorBoard
from keras import backend as K
from keras.layers import Lambda, Input, GlobalMaxPooling2D
from keras.callbacks import History 
K.set_image_data_format('channels_last')

def train_classifier(train_data, train_lbl, val_data, val_lbl, output_dir, tb_path, max_epochs, lr, batch_size, train_mode, lr_plateau =[5,0.01,1,0.000001], early_stop=[3,0.01], clf_dropout=0.2, restart_model=None, input_model=None, print_model_summary_only=False, use_resize=False, restart=False, histogram_graphs=False, instance_based=False, mean_pre_data=None):
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
    if restart_model is None:
        # add resize layer to fit images for InceptionResNetV2 input layer (299x299)
        if instance_based:
            model_1 = InceptionResNetV2(pooling='avg', input_shape=(256,256*2,3), weights =  "imagenet", include_top=False)     
            model_1.get_layer("conv_7b").kernel_regularizer = regularizers.l1(0.005)
            dropout_1 = layers.Dropout(clf_dropout,name='dropout_1')(model_1.output)

            # add classifier
            predictions = layers.Dense(num_classes, activation="softmax", name='predictions')(dropout_1)
            
            # create final model
            final_model = Model(input = model_1.input, output = predictions)

            if input_model is not None:
                final_model.load_weights(input_model, by_name=True)
        
        else:
            if use_resize:
                inp = Input(shape=(None, None, 3),name='image_input')
                inp_resize = Lambda(lambda image: K.tf.image.resize_images(image, (299, 299), K.tf.image.ResizeMethod.BICUBIC),name='image_resize')(inp)
                resize = Model(inp,inp_resize)
                # get the InceptionResNetV2 model and add it on top of the resize layer
                base_model = InceptionResNetV2(input_tensor=resize.output, pooling='avg', weights = "imagenet", include_top=False) 

            else: # use original image sizes
                # get the InceptionResNetV2 model
                base_model = InceptionResNetV2(weights = "imagenet", pooling='avg' , include_top=False, input_shape = (256, 256, 3))
            
            # create classifier - InceptionResNetV2 only uses an average pooling layer and a softmax classifier on top
            # Some articles mention that a dropout layer of 0.2 is used between the pooling layer and the softmax layer
            #base_model.get_layer("conv_7b").kernel_regularizer = regularizers.l1(0.01)
            clf = layers.Dropout(clf_dropout, name="dropout1")(base_model.output)
            predictions = layers.Dense(num_classes, activation="softmax" ,name="predictions")(clf)

            # Concatenate The Base And the Top Model
            final_model = Model(input = base_model.input, output = predictions)


    else:
        print("Using existing model: {}".format(restart_model))
        final_model = load_model(restart_model)

    if not restart:    
        # freeze all layers, so the trainable layers are controlled
        for layer in final_model.layers:
            if layer.name == "dropout1" or layer.name == "dropout_1" or layer.name == "dropout_2" :
                layer.rate = clf_dropout
                
            layer.trainable = False
        
    
    # data generators
    train_generator = DataGenerator(path_to_images=train_data,
                                    labels=cat_train_labels, 
                                    batch_size=batch_size,
                                    instance_based=instance_based,
                                    mean_sets=mean_pre_data)
    
    if histogram_graphs and not instance_based:

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
                                        batch_size=batch_size,
                                        instance_based=instance_based,
                                        mean_sets=mean_pre_data)
    
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
        flag = False
        if train_mode == "predictions":
            final_model.get_layer("predictions").trainable=True 
        elif train_mode == "mixed_7a":
            for layer in final_model.layers:
                if layer.name == "mixed_7a":
                    flag = True
                if flag:
                    layer.trainable = True
        else:
            for layer in final_model.layers:
                    layer.trainable = True
                
                
                
        # compile the model 
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True), metrics=["accuracy"])

    # print model summary and stop if specified
    final_model.summary()
    if print_model_summary_only:
        return 

    # fit model
    final_model.fit_generator(train_generator,
                        steps_per_epoch = None,
                        epochs = max_epochs,
                        validation_data = val_generator,
                        validation_steps = val_steps,
                        callbacks = callback_list,
                        workers=3,
                        use_multiprocessing=True)
    final_model.save(output_dir+"/final.h5")
    # print summary
    with open(output_dir + '/' + 'summary.txt','w') as fp:
        fp.write('Max epochs: '+ str(max_epochs)+'\n')
        fp.write('lr: '+ str(tensorboard.get_lr_hist())+'\n')
        fp.write('Batch size: '+ str(batch_size)+'\n')
        fp.write('Training mode: ' + str(train_mode)+'\n')
        fp.write('Training accuracy: ' + str(history.history['acc'])+'\n')
        fp.write('Validation accuracy: ' + str(history.history['val_acc'])+'\n')
    
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train specific layers of a InceptionResNetV2 keras model',
                                    description='''Train specific layers of a InceptionResNetV2 keras model. The model may be trained from scratch, pretrained or an existing model saved in an external file''')

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

    # only allow model to train whole inception "blocks"
    allowed_layers = ['predictions','full','mixed_7a']
    parser.add_argument('-train_mode', 
                        help='Layer to stop training from (layer is included in training). Limited to beginning of inception blocks',
                        required=True,
                        choices=allowed_layers)
    
    parser.add_argument('-inst_based', 
                        help='Instance Based Prediction Network',
                        type=str2bool,
                        default=False)

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
                        default=0.2)

    parser.add_argument('-batch_size', 
                        help='Batch size to use when training',
                        type=int,
                        default=32)
                        
    parser.add_argument('-restart_model', 
                        help='Path to .h5 model to train last layers',
                        default=None)

    parser.add_argument('-input_model', 
                        help='Path to .h5 model to initialize instance-based network',
                        default=None)                    
    
    parser.add_argument('-restart', 
                        help='Make sure that the model use loaded learn rate and architecture (model wont be compiled)',
                        type=str2bool,
                        default=False)

    parser.add_argument('-use_resize', 
                        help='Use resizing to 299*299*3',
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

    parser.add_argument('-mean_pre_data', 
                        help='Paths to all data sets that should be included in preprocessing',
                        nargs='*',
                        type=str,
                        required=False,
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
                        restart_model=args.restart_model, 
                        input_model=args.input_model,
                        train_mode=args.train_mode,
                        clf_dropout=args.dropout,
                        print_model_summary_only=args.summary_only,
                        use_resize=args.use_resize,
                        restart=args.restart,
                        early_stop=args.early_stop,
                        tb_path=args.tb_path,
                        histogram_graphs=args.histogram_graphs,
                        instance_based=args.inst_based,
                        mean_pre_data=args.mean_pre_data)