from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import  StratifiedKFold
import pandas as pd

from Knn import Knn
from distances.EuclideanDistance import EuclideanDistance
from distances.EuclideanDistance import *
from distances.ManhattanDistance import ManhattanDistance
from distances.CosineDistance import CosineDistance
from utils.preprocessor import prepare_dataset
from utils.dataset_splitter import split_dataset

experiment_dict = {
    'iris': {'name': 'Iris.csv', 'to_drop' : [], 'target': 'Species'},
    'diabetes': {'name': 'diabetes.csv', 'to_drop' : [], 'target': 'class'},
    'parkinson': {'name': 'parkinson.csv',  'to_drop' : [], 'target': 'status'},
    'epilepsy': {'name': 'epilepsy.csv', 'to_drop' : [], 'target': 'y'},
    'prostate': {'name': 'prostate_cancer.csv', 'to_drop' : [], 'target': 'diagnosis_result'}
}


def k_fold_cross_validation(X_train, Y_train, knn, dataset_name, n_folds):

    ## K-fold cross validation

    k_list = []
    accuracy_list = []
    precision_list = []
    f1_score_list = []

    k_fold_cross_validator = StratifiedKFold(n_splits=n_folds)
    epoch = 0

    for k in range(1, 7):

        accuracy_sum = 0
        precision_sum = 0
        f1_score_sum = 0

        for train_index, val_index in k_fold_cross_validator.split(X_train, Y_train):
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]
            prediction = knn.fit_knn_model(k, x_train, x_val, y_train)  # y_train is used to know the class to vote

            ## Compute metrics
            accuracy_sum += accuracy_score(y_val, prediction)
            precision_sum += precision_score(y_val, prediction, average='macro')
            f1_score_sum += f1_score(y_val, prediction, average='macro')

        ## Epoch completed, hyperparameters evaluated, compute average values over the folds
        k_list.append(k)
        accuracy_list.append(round(accuracy_sum / n_folds, 5))
        precision_list.append(round(precision_sum / n_folds, 5))
        f1_score_list.append(round(f1_score_sum / n_folds, 5))

        print("Epoch " + str(epoch) + " finished")
        epoch += 1

    ##Save final results on CSV file

    df = pd.DataFrame({'K': k_list, 'Accuracy': accuracy_list, 'Precision': precision_list, 'F1 score': f1_score_list})
    df.to_csv("tuning/tuning_" + dataset_name + "_" + knn.distance_metric.get_distance_name() + ".csv")


def final_evaluation_on_test(k, X_train, X_test, Y_train, Y_test, knn, dataset_name):

    prediction = knn.fit_knn_model(k, X_train, X_test, Y_train)  # y_train is used to know the class to vote

    ## Compute metrics
    accuracy_metric = accuracy_score(Y_test, prediction)
    precision_metric = precision_score(Y_test, prediction, average='macro')
    f1_score_metric = f1_score(Y_test, prediction, average='macro')

    ##Save final results on CSV file
    df = pd.DataFrame({'K': k, 'Accuracy': accuracy_metric, 'Precision': precision_metric, 'F1 score': f1_score_metric}, index = [1])
    df.to_csv("tuning/test_" + dataset_name + "_" + knn.distance_metric.get_distance_name() + ".csv")


def main():

    tuning = True

    ## Initial parameters

    dataset_name = 'iris'
    distance_metric = EuclideanDistance()

    ## Main

    knn = Knn(experiment_dict, dataset_name, distance_metric)
    X, Y = prepare_dataset(experiment_dict, dataset_name)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    if tuning:
        n_folds = 9
        k_fold_cross_validation(X_train, Y_train, knn, dataset_name, n_folds)
    else:
        final_evaluation_on_test(5, X_train, Y_test, Y_train, Y_test, knn, dataset_name)


if __name__ == "__main__":
    main()


