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
import re
import tensorflow as tf


def predict(test_data, model_path, output_dir):
    print("Running image-based predictions")     
    # Load data
    X = image_reader(test_data) * (1./255)
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()
    
    prob = final_model.predict(X,verbose=1)

    # prediction = highest probability (+offset since labels may not start at 0)
    prediction = np.argmax(prob, axis=1) + 1

    # save predictions model
    write_predictions_file(prediction,output_dir)


def instance_predict(test_data, model_path, output_dir, decision):
    print("Running non-augmented instance-based predictions")
    # Load data
    X_test = image_reader(test_data) * (1./255)
    num_test, h, w, c = X_test.shape
    
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()

    # predict
    prob_test = np.exp(final_model.predict(X_test,verbose=1))

    idx = 0
    prob = np.zeros((num_test,29))
    for img1, img2 in zip(prob_test[:-1:2], prob_test[1::2]):
        if decision == 'average':
            avg_prob = np.add(img1,img2)    # add probabilities (as we are only interested in the maximum, division is uncessary)
            prob[idx,:] = avg_prob
            idx += 1
            prob[idx,:] = avg_prob
            idx += 1
        elif decision == 'highest':
            img1_max = np.amax(img1)
            img2_max = np.amax(img2)

            if img1_max > img2_max:
                prob[idx,:] = img1
                idx += 1
                prob[idx,:] = img1
            else:
                prob[idx,:] = img2
                idx += 1
                prob[idx,:] = img2
            idx += 1         
    prediction = np.argmax(prob,axis=1) + 1
    
    # save predictions model
    write_predictions_file(prediction,output_dir)


def aug_instance_predict(test_data, aug_test_data, model_path, output_dir, decision):
    print("Running augmented instance-based predictions")
    # Load data
    X_test = image_reader(test_data) * (1./255)
    num_test, h, w, c = X_test.shape

    # load augmented test data
    is_root_dir = True
    extensions = ['jpg', 'jpeg']
    file_list = []
    dir_name = os.path.basename(aug_test_data)
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
        file_glob = os.path.join(aug_test_data, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')
        return

    aug_img_count = np.zeros(num_test)
    for image in file_list:
        image_num = int(re.search("(\d*\.?\d)",image).group(0))
        aug_img_count[image_num-1] += 1
    
    X_aug_test = image_reader(aug_test_data) * (1./255)
    
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()

    # predict
    prob_test = np.exp(final_model.predict(X_test,verbose=1))
    prob_aug_test = np.exp(final_model.predict(X_aug_test,verbose=1))

    prob = np.zeros((num_test,29))
    old_aug_idx = 0
    new_aug_idx = 0
    for idx in range(0,num_test,2):
        # image 1  
        img1 = prob_test[idx].reshape(1,29)                                 # extract original image
        num_aug1 = aug_img_count[idx]                                       # number of augmented samples
        new_aug_idx += num_aug1                                             # index of last augmented samples
        img1_aug_pred = prob_aug_test[int(old_aug_idx):int(new_aug_idx)]    # extract augmented samples
        old_aug_idx = new_aug_idx                                           # index of first augmented sample for next image

        # image 2
        img2 = prob_test[idx+1].reshape(1,29)                               # extract original image
        num_aug2 = aug_img_count[idx+1]                                     # number of augmented samples
        new_aug_idx += num_aug2                                             # index of last augmented samples
        img2_aug_pred = prob_aug_test[int(old_aug_idx):int(new_aug_idx)]    # extract augmented samples
        old_aug_idx = new_aug_idx                                           # index of first augmented sample for next image

        # concatenate augmented images and original images
        agg_prob = np.concatenate((img1, img2, img1_aug_pred, img2_aug_pred))
        
        if decision == 'average':
            sum_aug_prob = np.sum(agg_prob,axis=0)                  # sum probabilities across all columns (as we are only interested in the maximum, division is uncessary)
            prob[idx] = sum_aug_prob                                
            prob[idx+1] = sum_aug_prob
                
        elif decision == 'highest':
            max_vals = np.amax(agg_prob,axis=1)                     # find highest class probability for each sample
            prob[idx] = agg_prob[np.argmax(max_vals)]               # assign most confident prediction to both images
            prob[idx+1] = agg_prob[np.argmax(max_vals)]

        elif decision == 'weighted_average':
            max_vals = np.amax(agg_prob,axis=1)                     # find highest class probability for each sample
            weights = max_vals / np.amax(max_vals)                  # weigh the contribution of each sample, by its confidence in its prediction
            prob[idx] = np.dot(weights,agg_prob)                    # assign weighted sum of all sample predictions to both images
            prob[idx+1] = np.dot(weights,agg_prob)                  # (as we are only interested in the maximum, division is uncessary)

    prediction = np.argmax(prob,axis=1) + 1

    # save predictions model
    write_predictions_file(prediction,output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Predict test samples using specified classifier')

    parser.add_argument('-test_data', 
                        help='path to directory containing test images', 
                        nargs='?', 
                        default='Data/Test/TestImages')

    parser.add_argument('-aug_test_data', 
                        help='path to directory containing augmented test images', 
                        nargs='?', 
                        default='Data/Test/AugTestImages')

    parser.add_argument('-output', 
                        help='output file where results are stored',
                        required=True)

    parser.add_argument('-input_model', 
                        help='Path to .h5 model to make predictions',
                        required=True)

    parser.add_argument('-instance',
                        help='Use instanced-based classification',
                        action="store_true")

    parser.add_argument('-augmented',
                        help='Use augmented images to aid classification',
                        action="store_true")

    parser.add_argument('-decision_mode',
                        help='how instances are used to aid classification',
                        nargs=1,
                        choices=['average','highest','weighted_average'],
                        default=['average'])
    
    args = parser.parse_args()
    
    if args.instance:
        if args.augmented:
            aug_instance_predict(test_data=args.test_data,
                            aug_test_data=args.aug_test_data,
                            model_path=args.input_model,
                            output_dir=args.output,
                            decision=args.decision_mode[0])
        else:
            instance_predict(test_data=args.test_data,
                            model_path=args.input_model,
                            output_dir=args.output,
                            decision=args.decision_mode[0])
    else:
        predict(test_data=args.test_data, 
                model_path=args.input_model,
                output_dir=args.output)