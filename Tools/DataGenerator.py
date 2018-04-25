import numpy as np
from scipy.misc import imread
from Tools.ImageReader import create_image_lists
from keras.utils import Sequence


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, path_to_images, labels=None, shuffle=True, batch_size=32):
        'Initialization'
        # Store arguments
        self.batch_size = batch_size
        self.labels = labels.copy() if labels is not None else None
        # Get File names and labels locally
        self.img_list = create_image_lists(path_to_images)
        # Initialize Index To 0
        self.idx = 0
        # Store the length of available images
        self.num_images = len(self.img_list)
        self.shuffle = shuffle
        # Initialize First Epoch
        self.on_epoch_end()
   
    def on_epoch_end(self):
        self.idx=0
        if self.shuffle:
            'Updates image list after each epoch'
            # Random shuffle
            permutation = np.random.permutation(self.num_images)
            # Rearrange Images Given The Shuffle
            self.img_list = [self.img_list[i] for i in permutation]
            # Rearrange The Labels Given The Shuffle
            if self.labels is not None:
                self.labels = self.labels[permutation]
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.num_images / self.batch_size))

    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Get batch of indexes
        batch = np.array([ x % self.num_images for x in range(self.idx, self.idx + self.batch_size) ])
        # Update Index
        self.idx = (self.idx + self.batch_size) % self.num_images
        if self.idx > 0 and self.idx < self.batch_size-1:
            batch = batch[0:self.batch_size-self.idx]
        # Load images in numpy array
        X = np.array([np.array(imread(self.img_list[img_idx])) for img_idx in batch]) * (1. / 255)
        # Get the labels
        if self.labels is not None:
            y = self.labels[batch]
            return X, y
        else:
            return X

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        if self.labels is not None:
            X, y = self.__data_generation()
            return X, y
        else:
            X = self.__data_generation()
            return X



    


