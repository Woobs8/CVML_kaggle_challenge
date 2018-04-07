import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

class FullyConnectedClassifier:
    def __init__(self, hidden_layers=0, dimensions=None, momentum=0.9, batch_size=32, epochs=1000, dropout=0.5):
        # hyper parameters
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs

        # data parameters
        self.num_features = None
        self.num_samples = None
        self.num_classes = None
        self.classes = None

        # classifier
        self.model = Sequential()
        if hidden_layers >= 0:
            self.hidden_layers = hidden_layers
        else:
            self.hidden_layers = 0

        if dropout < 0:
            self.dropout = 0
        elif dropout > 1:
            self.dropout = 1
        else:
            self.dropout = dropout

        self.dimensions = dimensions
  

    def fit(self, train_data, train_labels, log_dir, lr_schedule, val_data=None, val_labels=None, class_weighting=True):
        self.num_samples, self.num_features = train_data.shape
        self.classes = list(set(train_labels))
        self.num_classes = len(self.classes)

        # labels must be from 0-num_classes-1, so label offset is subtracted
        self.label_offset = self.classes[0]
        train_labels -= self.label_offset
        if not val_labels is None:
            val_labels -= self.label_offset

        # determine class weights to account for difference in samples for classes
        if class_weighting:
            unique, counts = np.unique(train_labels, return_counts=True)
            class_weights = self.num_samples/counts
            normalized_class_weights = class_weights / np.max(class_weights)
            class_weights = dict(zip(unique, normalized_class_weights))
        else:
            class_weights = None

        # one-hot encode labels
        cat_train_labels = to_categorical(train_labels)
        cat_val_labels = to_categorical(val_labels)

        # generate list of layer dimensions
        if self.dimensions is None:
            dim_list = [self.num_features for x in range(0,self.hidden_layers)]
        else:
            dim_list = self.dimensions       

        prev_output_dim = self.num_features
        for idx, layer in enumerate(range(1,self.hidden_layers+1)):
            self.model.add(Dense(dim_list[idx], input_dim=prev_output_dim, activation='relu'))
            # add dropout between hidden layers
            if layer < self.hidden_layers:
                self.model.add(Dropout(self.dropout))
            prev_output_dim = dim_list[idx]
        self.model.add(Dense(self.num_classes, input_dim=prev_output_dim, activation='softmax'))

        # compile model
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=optimizers.SGD(lr=0.0, momentum=self.momentum, nesterov=True), 
                            metrics=['accuracy'])
        
        # define stopping criteria
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

        # define tensorboard callback
        log_path = os.path.join(log_dir,'Graph')
        tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)

        # fit the model
        if val_data is None or val_labels is None:
            hist = self.model.fit(train_data, cat_train_labels, validation_split=0.1, epochs=self.epochs, class_weight=class_weights, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])
        else:
            hist = self.model.fit(train_data, cat_train_labels, validation_data=(val_data,cat_val_labels), epochs=self.epochs, class_weight=class_weights, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])


        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_file = os.path.join(log_dir, 'fit_model.h5')
        self.model.save(model_file)  

        return hist.history


    def load(self, model_file, label_offset):
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        self.label_offset = label_offset

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
