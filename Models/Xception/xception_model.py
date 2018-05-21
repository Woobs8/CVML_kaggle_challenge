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

def xception_model(output_dir, input_model=None, print_model_summary=False, pretraining=True, use_resize=False, instance_based=False):
    if pretraining:
        print("Using pretraining")
        pretraining = "imagenet"
    else:
        print("Using randomly initiated weights")        
        pretraining = None

    #Instance Based (stiching 2 images together)
    if instance_based:
        print("Creating instance based model")
        base_model = Xception(pooling='avg', input_shape=(256,256*2,3), weights =  pretraining, include_top=False)     
    
    #Image Based with resizing (fit to original architecture input size)
    elif use_resize:
        print("Creating image based model using resize")
        inp = Input(shape=(None, None, 3),name='image_input')
        inp_resize = Lambda(lambda image: K.tf.image.resize_images(image, (299, 299), K.tf.image.ResizeMethod.BICUBIC),name='image_resize')(inp)
        resize = Model(inp,inp_resize)
        base_model = Xception(input_tensor=resize.output, pooling='avg', weights = pretraining, include_top=False) 

    # Image based input size is 256 x 256 x 3
    else:
        print("Creating image based model with input size 256x256x3")
        base_model = Xception(weights = pretraining, pooling='avg' , include_top=False, input_shape = (256, 256, 3))
    
    # Create Top Layers
    clf = layers.Dropout(0.2, name="dropout1")(base_model.output)
    predictions = layers.Dense(29, activation="softmax" ,name="predictions")(clf)

    # Concatenate The Base Model And The Top Model
    final_model = Model(input = base_model.input, output = predictions)

    # If input model is specified we load the model weights by name
    if input_model is not None:
        print("Using input model weights %s"%(input_model))    
        final_model.load_weights(input_model, by_name=True)
  
    # All layers are made trainable
    for layer in final_model.layers:
        layer.trainable = True
                
    # Compile the model 
    final_model.compile(loss = "categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=["accuracy"])

    # Print model summary
    if print_model_summary:
        final_model.summary()
    
    # Finish script
    final_model.save(output_dir+"/base_model.h5")
    print("Script finished model \"base_model.h5\" was placed in %s" %(output_dir))
    
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train specific layers of a InceptionResNetV2 keras model',
                                    description='''Train specific layers of a InceptionResNetV2 keras model. The model may be trained from scratch, pretrained or an existing model saved in an external file''')

    parser.add_argument('-output', 
                        help='output directory where results are stored',
                        required=True)

    parser.add_argument('-inst_based', 
                        help='Instance Based Prediction Network',
                        type=str2bool,
                        default=False)

    parser.add_argument('-pretraining', 
                        help='Boolean indicating wether to use pretrained imagenet weights',
                        type=str2bool,
                        default=True)

    parser.add_argument('-input_model', 
                        help='Path to .h5 model to initialize instance-based network',
                        default=None)                    
    
    parser.add_argument('-use_resize', 
                        help='Use resizing to 299*299*3',
                        type=str2bool,
                        default=False)

    parser.add_argument('-summary', 
                        help='Stop Script after prining model summary (ie. no training)',
                        type=str2bool,
                        default=False)

    args = parser.parse_args()
    
    xception_model( output_dir=args.output, 
                    input_model=args.input_model,
                    pretraining=args.pretraining,
                    print_model_summary=args.summary,
                    use_resize=args.use_resize,
                    instance_based=args.inst_based)