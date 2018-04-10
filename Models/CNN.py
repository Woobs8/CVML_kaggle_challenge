import numpy as np
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

SUPPORTED_ARCHITECTURES = ['VGG19']

class PretrainedConvolutionalNeuralNetwork:
    def __init__(self, architecture="VGG19", batch_size=32, epochs=1000, dropout=0.5,momentum=0.9, data_augmentation=False, num_freeze_layers=16, img_width = 256, img_height = 256, img_depth = 3):
        # hyper parameters
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_freeze_layers = num_freeze_layers
        self.data_augmentation = data_augmentation
        if dropout < 0:
            self.dropout = 0
        elif dropout > 1:
            self.dropout = 1
        else:
            self.dropout = dropout

        # data parameters
        self.img_depth = img_depth
        self.img_width = img_width
        self.img_height = img_height
        self.model = None

        # Architecture
        if architecture in SUPPORTED_ARCHITECTURES:
            self.architecture = architecture
        else:
            raise ValueError('Architecture is not supported')
        

    def fit(self, train_data, train_labels, steps_per_epoch, validation_steps, log_dir, lr_schedule, val_data=None, val_labels=None, class_weighting=True):

        # Get Unique Class Labels
        unique, counts = np.unique(train_labels, return_counts=True)
        num_samples = len(unique)
        
        # Load the architecture to be used
        if self.architecture is 'VGG19':
            self.model = self.create_vgg19_model(num_samples, self.img_width,self.img_height,self.img_depth,self.num_freeze_layers)
        #elif:
            #"""" MORE MODELS TO COME """
            # Insert New model
        else:
            raise ValueError('Architecture is not specified')

        # compile the model 
        self.model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0, momentum=self.momentum), metrics=["accuracy"])

        # labels must be from 0-num_classes-1, so label offset is subtracted
        self.label_offset = int(unique[0])
        train_labels -= self.label_offset
        if not val_labels is None:
            val_labels -= self.label_offset

        # determine class weights to account for difference in samples for classes
        if class_weighting:
            class_weights = num_samples/counts
            normalized_class_weights = class_weights / np.max(class_weights)
            class_weights = dict(zip(unique-self.label_offset, normalized_class_weights))
        else:
            class_weights = None

        # one-hot encode labels
        cat_train_labels = to_categorical(train_labels)
        cat_val_labels = to_categorical(val_labels)

        # Initiate the train and test generators
        if self.data_augmentation:
            train_datagen = ImageDataGenerator(
                rescale = 1./255,
                horizontal_flip = True,
                fill_mode = "nearest",
                zoom_range = 0.3,
                width_shift_range = 0.3,
                height_shift_range=0.3,
                rotation_range=30)
            
            train_generator = train_datagen.flow(
                train_data,
                cat_train_labels,
                batch_size = self.batch_size)

            validation_datagen = ImageDataGenerator(
                rescale = 1./255,
                horizontal_flip = True,
                fill_mode = "nearest",
                zoom_range = 0.3,
                width_shift_range = 0.3,
                height_shift_range=0.3,
                rotation_range=30)
            
            validation_generator = validation_datagen.flow(
                val_data,
                cat_val_labels,
                batch_size = self.batch_size)

        # Save the model according to the conditions  
        checkpoint = ModelCheckpoint(self.architecture+".h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

        # define tensorboard callback
        log_path = os.path.join(log_dir,'Graph')
        tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)

        if self.data_augmentation is True:
            # fit the model
            hist = self.model.fit_generator(
            train_generator,
            steps_per_epoch = len(train_data) / self.batch_size,
            epochs = self.epochs,
            validation_data = validation_generator,
            validation_steps = validation_steps,
            class_weight=class_weights,
            callbacks = [checkpoint, early, tensorboard, lr_schedule])
        else:
            if val_data is None or val_labels is None:
                hist = self.model.fit(train_data, cat_train_labels, validation_split=0.1, epochs=self.epochs, class_weight=class_weights, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])
            else:
                hist = self.model.fit(train_data, cat_train_labels, validation_data=(val_data,cat_val_labels), epochs=self.epochs, class_weight=class_weights, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])

        # Save The Model
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_file = os.path.join(log_dir, 'fit_model.h5')
        self.model.save(model_file)  

        return hist.history

    def create_vgg19_model(self, num_classes, img_width, img_height, img_depth, num_freeze_layers=16):
        # Number of layers to freeze must be between 0 and 16 layers
        if num_freeze_layers < 0:
            num_freeze_layers = 0
        elif num_freeze_layers > 16:
            num_freeze_layers = 16

        # Get The VGG19 Model
        model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, img_depth))
        """
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         (None, 256, 256, 3)       0         
        _________________________________________________________________
        block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
        _________________________________________________________________
        block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
        _________________________________________________________________
        block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
        _________________________________________________________________
        block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
        _________________________________________________________________
        block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
        _________________________________________________________________
        block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
        _________________________________________________________________
        block3_conv1 (Conv2D)        (None, 64, 64, 256)       295168    
        _________________________________________________________________
        block3_conv2 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_conv3 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_conv4 (Conv2D)        (None, 64, 64, 256)       590080    
        _________________________________________________________________
        block3_pool (MaxPooling2D)   (None, 32, 32, 256)       0         
        _________________________________________________________________
        block4_conv1 (Conv2D)        (None, 32, 32, 512)       1180160   
        _________________________________________________________________
        block4_conv2 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_conv3 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_conv4 (Conv2D)        (None, 32, 32, 512)       2359808   
        _________________________________________________________________
        block4_pool (MaxPooling2D)   (None, 16, 16, 512)       0         
        _________________________________________________________________
        block5_conv1 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_conv4 (Conv2D)        (None, 16, 16, 512)       2359808   
        _________________________________________________________________
        block5_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
        =================================================================
        Total params: 20,024,384.0
        Trainable params: 20,024,384.0
        Non-trainable params: 0.0
        """

        # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
        for layer in model.layers[:num_freeze_layers]:
            layer.trainable = False

        #Adding custom Layers 
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(num_classes, activation="softmax")(x)

        # creating the final model 
        model_final = Model(input = model.input, output = predictions)

        return model_final


    def load(self, model_file):
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        return self

    def predict(self, data, output_file=""):   
        prob = self.model.predict(data)

        # prediction = highest probability (+offset since labels may not start at 0)
        prediction = np.argmax(prob,axis=1)+self.label_offset

        if output_file != "":
            dir = os.path.dirname(output_file)
            if dir != "" and not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(output_file, prediction)

        return prediction

