import numpy as np
from sklearn.svm import SVC, LinearSVC
import os
from sklearn.externals import joblib

class SupportVectorMachine:
    def __init__(self, kernel='linear', C=100, gamma='auto', max_iter=100, coef0=0, tol=1e-3, degree=3):
        # Hyperparameters
        self.kernel = kernel
        self.max_iter = max_iter
        self.gamma = gamma
        self.C = C
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
            self.clf=LinearSVC(max_iter=max_iter, C=C, tol=tol, class_weight='balanced')
        elif kernel=='rbf' or kernel=='poly' or kernel=='sigmoid':
            self.clf=SVC(kernel=kernel, C=C, degree=degree,gamma=gamma,max_iter=max_iter, coef0=coef0,tol=tol,class_weight='balanced', probability=True)
        else:
            print(str(self.__class__)+" Error: unknown kernel '"+kernel+"'")
            exit(1)
        

    def fit(self, data, labels, model_file=""):
        self.num_samples, self.num_features = data.shape
        self.classes = list(set(labels))
        self.num_classes = len(self.classes)

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


    def load(self, model_file):
        if os.path.exists(model_file):
            self.clf = joblib.load(model_file)  

        return self  


    def predict(self, data, output_file="", instance=False, decision='average'):
        num_test, num_features = data.shape
        if instance:
            prob_test = self.clf.predict_proba(data)

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
        else:
            prediction = self.clf.predict(data)

        if output_file != "":
            dir = os.path.dirname(output_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.savetxt(output_file, prediction)

        return prediction
