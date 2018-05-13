#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
import argparse
import numpy as np
from keras.models import Model, load_model
from Tools.DataWriter import write_predictions_file
from Tools.ImageReader import image_reader
from Tools.DataGenerator import DataGenerator
from keras import backend as K
import re
import tensorflow as tf


def predict(test_data, model_path, output_dir,mean_pre_data):
    print("Running image-based predictions")     
    # Load data
    #X = image_reader(test_data) * (1./255)
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()
    
    batch_size = 32
    pred_gen = DataGenerator(   path_to_images=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                use_augment=False,
                                mean_sets=mean_pre_data)
    prob = []
    num_steps = pred_gen.__len__()
    for idx in range(num_steps):
        X_batch, id_batch = pred_gen.__getitem__(idx)
        prob.append(np.exp(final_model.predict_on_batch(X_batch)))
    prob = np.vstack(prob)

    # prediction = highest probability (+offset since labels may not start at 0)
    prediction = np.argmax(prob, axis=1) + 1

    # save predictions model
    write_predictions_file(prediction,output_dir)


def instance_predict(test_data, model_path, output_dir, decision, dual_mode,mean_pre_data):
    print("Running non-augmented instance-based predictions")
    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()
    
    # Data Generator
    batch_size = 32
    pred_gen = DataGenerator(   path_to_images=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                use_augment=False,
                                instance_based=dual_mode,
                                mean_sets=mean_pre_data)

    # predict
    prob_test = []
    id_prob = []
    num_test = pred_gen.num_images
    num_steps = pred_gen.__len__()
    for idx in range(num_steps):
        X_batch, id_batch = pred_gen.__getitem__(idx)
        prob_test.append(np.exp(final_model.predict_on_batch(X_batch)))
        id_prob.append(id_batch)
        
        # Print Progress
        factor1 = int(np.ceil(idx*1./(num_steps-1) * 30))
        factor2 = int(np.floor((1-idx*1./(num_steps-1)) * 30))
        print_string = 'Step '+ str(idx+1) + ' / ' + str(num_steps) + ' : [' + '=' * factor1 + '>' + ' ' * factor2+ ']'
        if idx < num_steps-1:
            print(print_string, end="\r", flush=True)
        else:
            print(print_string, flush=True)

    prob_test = np.vstack(prob_test)
    id_prob = np.hstack(id_prob)
    prob = np.zeros((num_test,29))
    if not dual_mode:
        id_prob = id_prob - id_prob % 2 # Instance based every 2 images is the same sample. Rename the sample ids to [0 0 2 2 4 4 ...] from [0 1 2 3 4 5 ...]

        prob = np.zeros((num_test,29))
        arg_max_rows = []
        for idx in np.unique(id_prob):
            if decision == 'average':
                sum_prob = np.sum(prob_test[np.where(id_prob==idx)[0]],axis=0)  # sum probabilities across all columns (as we are only interested in the maximum, division is uncessary)
                prob[idx] = sum_prob
                prob[idx+1] = sum_prob
            elif decision == 'highest':
                index = np.unravel_index(np.argmax(prob_test[np.where(id_prob==idx)[0]], axis=None), prob_test.shape)   # find highest class probability
                prob[idx,index[1]] = prob_test[index]                                                      #  assign most confident prediction to both images
                prob[idx+1,index[1]] = prob_test[index]  
    else:
        for idx in np.unique(id_prob):
            index = np.unravel_index(np.argmax(prob_test[np.where(id_prob==idx)[0]], axis=None), prob_test.shape)   # find highest class probability
            prob[idx,index[1]] = prob_test[index]                                                      #  assign most confident prediction to both images
            prob[idx+1,index[1]] = prob_test[index]   
    
    prediction = np.argmax(prob, axis=1) + 1
    
    # save predictions model
    write_predictions_file(prediction,output_dir)


def aug_instance_predict(test_data, aug_test_data, model_path, output_dir, decision, dual_mode, num_aug,mean_pre_data):
    print("Running augmented instance-based predictions")

    # load pre-trained model
    final_model = load_model(model_path)
    final_model.summary()

    # Data Generator original images
    batch_size_org = 8
    pred_gen_org = DataGenerator(   path_to_images=test_data,
                                    batch_size=batch_size_org,
                                    shuffle=False,
                                    use_augment=False,
                                    instance_based=dual_mode,
                                    mean_sets=mean_pre_data)

    # Data Generator augmented images
    batch_size_aug = 8
    pred_gen_aug = DataGenerator(   path_to_images=test_data,
                                    batch_size=batch_size_aug,
                                    shuffle=False,
                                    use_augment=True,
                                    instance_based=dual_mode,
                                    predict_aug_size=num_aug,
                                    mean_sets=mean_pre_data)
    # predict original images
    prob_test = []
    id_org = []
    num_test = pred_gen_org.num_images
    num_steps = pred_gen_org.__len__()

    for idx in range(num_steps):
        X_batch, id_batch = pred_gen_org.__getitem__(idx)
        prob_test.append(np.exp(final_model.predict_on_batch(X_batch)))
        id_org.append(id_batch)
        # Print Progress
        factor1 = int(np.ceil(idx*1./(num_steps-1) * 30))
        factor2 = int(np.floor((1-idx*1./(num_steps-1)) * 30))
        print_string = 'Step '+ str(idx+1) + ' / ' + str(num_steps) + ' : [' + '=' * factor1 + '>' + ' ' * factor2+ ']'
        if idx < num_steps-1:
            print(print_string, end="\r", flush=True)
        else:
            print(print_string, flush=True)

    prob_test = np.vstack(prob_test)
    id_org = np.hstack(id_org)

    # predict on augmented images
    prob_aug_test = []
    id_aug = []
    num_steps = pred_gen_aug.__len__()
    for idx in range(num_steps):
        X_batch, id_batch = pred_gen_aug.__getitem__(idx)
        prob_aug_test.append(np.exp(final_model.predict_on_batch(X_batch)))
        id_aug.append(id_batch)
        # Print Progress
        factor1 = int(np.ceil(idx*1./(num_steps-1) * 30))
        factor2 = int(np.floor((1-idx*1./(num_steps-1)) * 30))
        print_string = 'Step '+ str(idx+1) + ' / ' + str(num_steps) + ' : [' + '=' * factor1 + '>' + ' ' * factor2+ ']'
        if idx < num_steps-1:
            print(print_string, end="\r", flush=True)
        else:
            print(print_string, flush=True)
    prob_aug_test = np.vstack(prob_aug_test)
    id_aug = np.hstack(id_aug)
    
    # Concatenate original image preds and augmented image preds
    agg_prob = np.concatenate((prob_test,prob_aug_test),axis = 0)

    # Concatenate IDs  
    if not dual_mode:
        id_agg_prob = np.concatenate((id_org,id_aug))
        id_agg_prob = id_agg_prob - id_agg_prob % 2 # Instance based every 2 images is the same sample. Rename the sample ids to [0 0 2 2 4 4 ...] from [0 1 2 3 4 5ª ...]
    else:
        id_agg_prob = np.concatenate((id_org,id_aug))
    

    # Calculate probabilites based on chosen method
    prob = np.zeros((num_test,29))
    for idx in np.unique(id_agg_prob):
        if decision == 'average':
            sum_aug_prob = np.sum(agg_prob[np.where(id_agg_prob==idx)[0]],axis=0)  # sum probabilities across all columns (as we are only interested in the maximum, division is uncessary)
            prob[idx] = sum_aug_prob                                
            prob[idx+1] = sum_aug_prob
                
        elif decision == 'highest':
            index = np.unravel_index(np.argmax(agg_prob[np.where(id_agg_prob==idx)[0]], axis=None), prob_test.shape)   # find highest class probability
            prob[idx,index[1]] = agg_prob[index]                                                      #  assign most confident prediction to both images
            prob[idx+1,index[1]] = agg_prob[index]  

        elif decision == 'weighted_average':
            max_vals = np.amax(agg_prob[np.where(id_agg_prob==idx)[0]],axis=1)  # find highest class probability for each sample
            weights = max_vals / np.amax(max_vals)                              # weigh the contribution of each sample, by its confidence in its prediction
            prob[idx] = np.dot(weights,agg_prob[np.where(id_agg_prob==idx)[0]])                                # assign weighted sum of all sample predictions to both images
            prob[idx+1] = np.dot(weights,agg_prob[np.where(id_agg_prob==idx)[0]])                              # (as we are only interested in the maximum, division is uncessary)
    
    # Calculate the actual predicted label    
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

    parser.add_argument('-num_aug',
                        help='Number of augmented images for each original Image',
                        type=int,
                        default=5)

    parser.add_argument('-dual_mode',
                        help='Model to use must recieve \'stiched\' images (256*512*3)',
                        action="store_true")

    parser.add_argument('-decision_mode',
                        help='how instances are used to aid classification',
                        nargs=1,
                        choices=['average','highest','weighted_average'],
                        default=['average'])
    parser.add_argument('-mean_pre_data', 
                        help='Paths to all data sets that should be included in preprocessing',
                        nargs='*',
                        type=str,
                        required=False,
                        default=False)
    
    args = parser.parse_args()
    if args.instance:
        if args.augmented:
            aug_instance_predict(test_data=args.test_data,
                            aug_test_data=args.aug_test_data,
                            model_path=args.input_model,
                            output_dir=args.output,
                            decision=args.decision_mode[0],
                            dual_mode=args.dual_mode,
                            num_aug=args.num_aug,
                            mean_pre_data=args.mean_pre_data)
        else:
            instance_predict(test_data=args.test_data,
                            model_path=args.input_model,
                            output_dir=args.output,
                            decision=args.decision_mode[0],
                            dual_mode=args.dual_mode,
                            mean_pre_data=args.mean_pre_data)
    else:
        predict(test_data=args.test_data, 
                model_path=args.input_model,
                output_dir=args.output,
                mean_pre_data=args.mean_pre_data)
