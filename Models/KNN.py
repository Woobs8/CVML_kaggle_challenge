from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.externals import joblib
import numpy as np


class KNearestNeighbors:
    def __init__(self, neighbor_count, neighbor_weight='distance', n_jobs=1):
        self.clf = KNeighborsClassifier(neighbor_count, weights=neighbor_weight, n_jobs=n_jobs)
        self.neighbors = neighbor_count
        self.neighbor_weights = neighbor_weight


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