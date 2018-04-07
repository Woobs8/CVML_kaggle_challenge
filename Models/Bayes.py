import numpy as np
from sklearn.naive_bayes import  GaussianNB as NB
import os
from sklearn.externals import joblib

class NaiveBayesClassifier:
    def __init__(self):
        self.clf = NB()


    def fit(self, data, labels, model_file=""):
        self.clf.fit(data, labels)

        # save model
        if model_file != "":
            dir = os.path.dirname(model_file)
            if dir != "" and not os.path.exists(dir):
                os.makedirs(dir)
            joblib.dump(self.clf,model_file)
        return self


    def predict(self, data, output_file=""):
        prediction = self.clf.predict(data)

        if output_file != "":
            dir = os.path.dirname(output_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(output_file, prediction)

        return prediction