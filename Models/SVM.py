import numpy as np
from sklearn.svm import SVC, LinearSVC
import os
from sklearn.externals import joblib

class SupportVectorMachine:
    def __init__(self, kernel='linear', gamma='auto', max_iter=100, coef0=0, tol=1e-3, degree=3):
        # Hyperparameters
        self.kernel = kernel
        self.max_iter = max_iter
        self.gamma=gamma
        self.coef0 = coef0
        self.tol = tol
        self.degree = degree
        self.dual = True

        # Data parameters
        self.num_features = None
        self.num_samples = None
        self.num_classes = None
        self.classes = None

        # Classifier
        if kernel=='linear':
            self.clf=LinearSVC(max_iter=max_iter,tol=tol)
        elif kernel=='rbf' or kernel=='poly' or kernel=='sigmoid':
            self.clf=SVC(kernel=kernel,degree=degree,gamma=gamma,max_iter=max_iter, coef0=coef0,tol=tol)
        else:
            print(str(self.__class__)+" Error: unknown kernel '"+kernel+"'")
            exit(1)
        

    def fit(self, data, labels, model_file=""):
        self.num_samples, self.num_features = data.shape
        self.classes = list(set(labels))
        num_classes = len(self.classes)

        if self.kernel=='linear' and (self.num_samples > self.num_features):
            self.dual = False
            self.clf.set_params(dual=self.dual)
    
        self.clf.fit(data,labels)

        # save model
        if model_file != "":
            dir = os.path.dirname(model_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            joblib.dump(self.clf,model_file)

        return self


    def predict(self, data, model_file="", output_file=""):
        if model_file != "" and os.path.exists(model_file):
            self.clf = joblib.load(model_file)

        prediction = self.clf.predict(data)

        if output_file != "":
            dir = os.path.dirname(output_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(output_file, prediction)

        return prediction
