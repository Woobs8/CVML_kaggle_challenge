import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..','..'))
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from Tools.DataReader import load_vector_data
import argparse

def main(k, threads, norm):
    training_data_path = 'Data/Train/trainVectors.txt'
    training_lbls_path = 'Data/Train/trainLbls.txt'
    validation_data_path = 'Data/Validation/valVectors.txt'
    validation_lbls_path = 'Data/Validation/valLbls.txt'

    # Loading dataset
    train_data, train_labels = load_vector_data(training_data_path, training_lbls_path)
    val_data, val_labels = load_vector_data(validation_data_path, validation_lbls_path)

    # Set the parameters by cross-validation
    tuned_parameters = [{'n_neighbors': k, 'weights':['uniform','distance']}]

    scores = ['accuracy']

    # normalize data
    if norm:
        print("############# Normalizing data #############")
        train_data = normalize(train_data)
        val_data = normalize(val_data)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5,
                        scoring='%s' % score, n_jobs=threads)
        clf.fit(train_data, train_labels)

        print("Best parameters set found on normalized development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = val_labels, clf.predict(val_data)
        print(classification_report(y_true, y_pred))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run grid search sklearn KNN')

    parser.add_argument('-neighbors',
                        nargs='*',
                        type=int)

    parser.add_argument('-threads',
                        nargs=1,
                        type=int,
                        default=1)

    args = parser.parse_args()
    
    main(k=args.neighbors,
        threads=args.threads[0],
        norm=True)

    main(k=args.neighbors,
        threads=args.threads[0],
        norm=False)