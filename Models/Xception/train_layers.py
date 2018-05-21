#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras import optimizers,layers,regularizers
from keras.models import Model, load_model, clone_model
from keras.utils import to_categorical
from keras.applications import Xception
from Tools.DataGenerator import DataGenerator
from Tools.DataReader import load_labels
from Tools.ImageReader import image_reader
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from Tools.tensorboard_lr import LRTensorBoard
from keras import backend as K
from keras.layers import Lambda, Input, GlobalMaxPooling2D
from keras.callbacks import History 
K.set_image_data_format('channels_last')
from sklearn.utils.class_weight import compute_class_weight

def train_classifier(train_data, train_lbl, val_data, val_lbl, output_dir, model, tb_path, train_mode, max_epochs, lr, batch_size, clf_dropout, use_augment=True, lr_plateau =[5,0.01,1,0.000001], early_stop=[3,0.01], print_model_summary_only=False, restart=False, histogram_graphs=False, mean_pre_data=None):
    # load labels
    training_labels = load_labels(train_lbl)
    validation_labels = load_labels(val_lbl)
    class_weights = compute_class_weight('balanced', np.unique(training_labels), training_labels)
    
    # labels must be from 0-num_classes-1, so label offset is subtracted
    unique, count = np.unique(training_labels,return_counts=True) 
    num_classes = len(unique)
    label_offset = int(unique[0])
    training_labels -= label_offset
    validation_labels -= label_offset

    # one-hot encode labels
    cat_train_labels = to_categorical(training_labels)
    cat_val_labels = to_categorical(validation_labels)
    
    # load model
    final_model = load_model(model)

    # control dropout
    if not restart:    
        for layer in final_model.layers:
            if "dropout" in layer.name:
                layer.rate = clf_dropout

    # Determine if model is instance based
    if final_model.input.shape[1] != final_model.input.shape[2]:
        instance_based = True
    else:
        instance_based = False
    
    # data generators
    train_generator = DataGenerator(path_to_images=train_data,
                                    labels=cat_train_labels, 
                                    batch_size=batch_size,
                                    instance_based=instance_based,
                                    mean_sets=mean_pre_data,
                                    use_augment=use_augment)
    
    # If we want histogram graphs we must pass all val images as numpy array
    if histogram_graphs:
        hist_frq = 1
        validation_images = image_reader(val_data)*(1./255)
        val_generator = (validation_images, cat_val_labels)
        val_steps = None
    else:
        hist_frq = 0
        val_steps = len(validation_labels)/64
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
    
    # If we are not to restart training from last point, set trainable layers
    if not restart:
        flag = False
        for layer in final_model.layers:
            if train_mode == "top":
                if "fc" in layer.name or "pred" in layer.name:
                    layer.trainable=True
                else: 
                    layer.trainable=False
            elif train_mode == "all":
                layer.trainable = True
            else:
                layer.trainable = False
             
        # compile the model 
        final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["accuracy"])

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
                        use_multiprocessing=True,
                        class_weight=class_weights)

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

    parser.add_argument('-train_mode', 
                        help='Layer to stop training from (layer is included in training). Limited to beginning of inception blocks',
                        required=True,
                        choices=['all','top'])
                        
    parser.add_argument('-model', 
                        help='Path to .h5 model to initialize instance-based network',
                        default=None,
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
                        default=0.2)

    parser.add_argument('-batch_size', 
                        help='Batch size to use when training',
                        type=int,
                        default=32)

    parser.add_argument('-use_augment', 
                        help='Make sure that the model use loaded learn rate and architecture (model wont be compiled)',
                        type=str2bool,
                        default=True)                  
    
    parser.add_argument('-restart', 
                        help='Make sure that the model use loaded learn rate and architecture (model wont be compiled)',
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
                        model=args.model,
                        train_mode=args.train_mode,
                        clf_dropout=args.dropout,
                        use_augment=args.use_augment,
                        print_model_summary_only=args.summary_only,
                        restart=args.restart,
                        early_stop=args.early_stop,
                        tb_path=args.tb_path,
                        histogram_graphs=args.histogram_graphs,
                        mean_pre_data=args.mean_pre_data)