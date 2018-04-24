from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
import numpy as np

class LRTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', **kwargs):
        self.lr_hist = []
        super(LRTensorBoard, self).__init__(log_dir, **kwargs)

    def set_model(self, model):
        super(LRTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Value to inject in logs must be a 1 element numpy array!
        learn_rate = K.get_value(self.model.optimizer.lr).copy()
        # Save the learn rate in the hist list
        self.lr_hist.append(float(learn_rate))
        # Inject Learn Rate
        logs['learn_rate'] = learn_rate

        super(LRTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(LRTensorBoard, self).on_train_end(logs)

    def get_lr_hist(self):
        return self.lr_hist