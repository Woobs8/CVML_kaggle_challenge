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
    prob_test = final_model.predict(X_test,verbose=1)

    idx = 0
    prob = np.zeros((num_test,29))
    for img1, img2 in zip(prob_test[:-1:2], prob_test[1::2]):
        if decision == 'average':
            avg_prob = np.add(img1,img2) / 2
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


def aug_instance_predict(test_data, aug_test_data, model_path, num_aug, output_dir,decision):
    print("Running augmented instanced-based predictions")
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
    prob_test = final_model.predict(X_test,verbose=1)
    prob_aug_test = final_model.predict(X_aug_test,verbose=1)

    prediction = np.zeros(num_test)
    if decision == 'average':
        for idx, pred in enumerate(prob_test):
            sum_aug_pred = np.sum(prob_aug_test[idx*num_aug:((idx*num_aug)+num_aug)])
            agg_prob = np.add(pred,sum_aug_pred)
            prob = agg_prob / (num_aug+1)
                
            # prediction = highest probability (+offset since labels may not start at 0)
            prediction[idx] = np.argmax(prob, axis=0) + 1
    elif decision == 'highest':
        cur_idx = 0
        old_idx = 0
        for idx, img in enumerate(zip(prob_test,aug_img_count)):
            cur_idx = old_idx + img[1]
            pred = img[0]
            aug_pred = prob_aug_test[int(old_idx):int(cur_idx)]
            pred = pred.reshape(1,29)
            agg_preds = np.append(aug_pred,pred,axis=0)
            max_vals = np.empty(int(img[1]+1),dtype='int64')
            max_idcs = np.argmax(agg_preds, axis=1, out=max_vals)
            pred_idx = np.argmax(max_vals)
            prediction[idx] = (max_idcs[pred_idx] % 29) + 1 
            old_idx = cur_idx
    
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
                        help='Used instanced-based classification',
                        action="store_true")

    parser.add_argument('-instance_count',
                        help='Number of augmented instances of test images to aid prediction',
                        nargs=1,
                        type=int,
                        default=None)

    parser.add_argument('-decision_mode',
                        help='how instances are used to aid classification',
                        nargs=1,
                        choices=['average','highest'],
                        default='highest')
    
    args = parser.parse_args()
    
    if args.instance:
        if args.instance_count is not None:
            aug_instance_predict(test_data=args.test_data,
                            aug_test_data=args.aug_test_data,
                            model_path=args.input_model,
                            num_aug=args.instance_count[0],
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