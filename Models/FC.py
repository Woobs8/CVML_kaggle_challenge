import numpy as np
import os
from Tools.DataLoader import load_vector_data
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
  

    def fit(self, train_data, train_labels, log_dir, lr_schedule, val_data=None, val_labels=None):
        self.num_samples, self.num_features = train_data.shape
        self.classes = list(set(train_labels))
        self.num_classes = len(self.classes)

        # one-hot encode labels (labels start at 1, so ignore first column)
        cat_train_labels = to_categorical(train_labels)[:,1:]
        cat_val_labels = to_categorical(val_labels)[:,1:]

        # generate list of layer dimensions
        if self.dimensions is None:
            dim_list = [self.num_features for x in range(0,self.hidden_layers)]
        else:
            dim_list = self.dimensions       

        prev_output_dim = self.num_features
        for idx, i in enumerate(range(0,self.hidden_layers)):
            self.model.add(Dense(dim_list[idx], input_dim=prev_output_dim, activation='relu'))
            self.model.add(Dropout(self.dropout))
            prev_output_dim = dim_list[idx]
        self.model.add(Dense(self.num_classes, input_dim=prev_output_dim, activation='softmax'))

        # compile model
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=optimizers.SGD(lr=0.0, momentum=self.momentum, nesterov=True), 
                            metrics=['accuracy'])
        
        # define stopping criteria
        early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')

        # define tensorboard callback
        log_path = os.path.join(log_dir,'Graph')
        tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)

        # fit the model
        if val_data is None or val_labels is None:
            hist = self.model.fit(train_data, cat_train_labels, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])
        else:
            hist = self.model.fit(train_data, cat_train_labels, validation_data=(val_data,cat_val_labels), epochs=self.epochs, batch_size=self.batch_size, callbacks=[early,tensorboard,lr_schedule])


        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_file = os.path.join(log_dir, 'fit_model.h5')
        self.model.save(model_file)  

        return hist.history


    def load(self, model_file):
        if os.path.exists(model_file):
            self.model = load_model(model_file)

        return self


    def predict(self, data, output_file=""):      
        prob = self.model.predict(data)

        # prediction = highest probability (+1 since labels start at 1)
        prediction = np.argmax(prob,axis=1)+1

        if output_file != "":
            dir = os.path.dirname(output_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(output_file, prediction)

        return prediction
