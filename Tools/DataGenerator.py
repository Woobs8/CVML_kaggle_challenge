import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scipy.misc import imread
from Tools.ImageReader import create_image_lists
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, path_to_images, labels=None, shuffle=True, batch_size=32, use_augment=False, instance_based=False,predict_aug_size=1):
        'Initialization'
        # Store arguments
        if instance_based:
            self.batch_size = batch_size * 2
        else:
            self.batch_size = batch_size
        # Get File names and labels locally
        self.img_list = np.array(create_image_lists(path_to_images))
        # Store the length of available images
        self.num_images = len(self.img_list)
        # Store Labels
        if labels == None: # return original index instead
            self.labels = np.arange(self.num_images)
            # Array to store the returned images ID (idx) when predicting
        else:
            self.labels = labels
        
        # Initialize Index To 0
        self.idx = 0
        # Shuffle
        self.shuffle = shuffle
        # Current arrangement of samples
        self.permutation = np.arange(self.num_images)
        
        # Data Generator
        if use_augment:
            self.datagen = ImageDataGenerator(  
                rescale = 1./255,
                horizontal_flip = True,
                vertical_flip=True,
                fill_mode = "nearest",
                rotation_range=50,
                #featurewise_center=True,
                data_format="channels_last")
        else:# Always rescale
            self.datagen = ImageDataGenerator(  
                rescale = 1./255,
                data_format="channels_last")
        
        # Instance based generator
        self.instance_based = instance_based
        # Predic Aug
        if predict_aug_size < 1:
            self.predict_aug_size = 1
        self.predict_aug_size = predict_aug_size
        # Initialize First Epoch
        self.on_epoch_end()
    
    def on_epoch_end(self):
        'Updates image list after each epoch'
        if self.shuffle:
            # Random shuffle
            if not self.instance_based:
                self.permutation = np.random.permutation(self.num_images)
            else:
                inst_perm = np.random.permutation(int(self.num_images/2))
                rep_perm = np.repeat(inst_perm,2)*2 # times 2 because we only have half the values
                rep_perm[1::2] += 1 # Every second repeated value should be added 1 to include the pair image
                self.permutation = rep_perm
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.num_images / self.batch_size))


    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Get batch of indexes
        batch = np.array([ x % self.num_images for x in range(self.idx, self.idx + self.batch_size) ])
        # Update Index
        self.idx = (self.idx + self.batch_size) % self.num_images
        if self.idx > 0 and self.idx < self.batch_size - 1:
            self.idx = 0
            batch = batch[0:self.batch_size-self.idx]
        # Current images in numpy array
        curr_list = self.img_list[self.permutation]
        X = np.array([np.array(imread(curr_list[img_idx])) for img_idx in batch])
        # Get the labels
        curr_labels = self.labels[self.permutation]
        y = curr_labels[batch]
        # Augmentation
        X, y = self.__augment_images__(X,y)
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()
        return X, y
    
    def set_shuffle(self,shuffle):
        self.shuffle = shuffle

    def __augment_images__(self,X,y):
        X_batch = []
        y_batch = []
        idx = 0
        for X_tmp, y_tmp in self.datagen.flow(X, y, 
                                        batch_size=self.batch_size,
                                        shuffle=False):
            X_batch.append(X_tmp) 
            y_batch.append(y_tmp)
            idx+=1
            if idx >= self.predict_aug_size:
                break
        X_batch=np.vstack(X_batch)
        y_batch=np.hstack(y_batch)
        if self.instance_based:
            if len(X_batch) % 2 != 0:
                raise ValueError("Batch size must be even numbered to use instance based")
            perm = np.random.permutation(2)
            X_batch = np.array([np.concatenate((X_batch[2*idx+perm[0]],X_batch[2*idx+perm[1]]),axis=1) for idx in range(int(len(X_batch) / 2))])
            y_batch = y_batch[::2]

        return X_batch, y_batch