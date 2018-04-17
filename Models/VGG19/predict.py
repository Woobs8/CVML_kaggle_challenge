#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras.models import Model, load_model
from Tools.DataWriter import write_predictions_file
from Tools.ImageReader import image_reader
from keras import backend as K


def predict(test_data, model_path, output_dir):     

    # Load data
    X = image_reader(test_data)
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()
    
    prob = final_model.predict(X,verbose=1)

    # prediction = highest probability (+offset since labels may not start at 0)
    prediction = np.argmax(prob, axis=1) + 1

    # save predictions model
    write_predictions_file(prediction,output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Fine-tune pre-trained network and existing classifier',
                                    description='''Train last layers of pretrained convolutional neural network and store the output to the specified directory''')

    parser.add_argument('test_data', 
                        help='path to directory containing test images', 
                        nargs='?', 
                        default='Data/Test/')

    parser.add_argument('-output', 
                        help='output directory where results are stored',
                        required=True)

    parser.add_argument('-input_model', 
                        help='Path to .h5 model to make predictions',
                        required=True)
    

    args = parser.parse_args()
    
    predict(test_data=args.test_data, 
            model_path=args.input_model,
            output_dir=args.output)