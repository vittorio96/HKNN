from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import  StratifiedKFold
import pandas as pd

from Knn import Knn
from distances.EuclideanDistance import *
from distances.ManhattanDistance import *
from distances.CosineDistance import *
from distances.EuclideanCosineHybridDistance import EuclideanCosineHybridDistance
from distances.EuclideanManhattanHybridDistance import EuclideanManhattanHybridDistance
from distances.CosineManhattanHybridDistance import CosineManhattanHybridDistance
from utils.preprocessor import prepare_dataset
from utils.dataset_splitter import split_dataset

experiment_dict = {
    'iris': {'name': 'Iris.csv', 'to_drop' : [], 'target': 'Species', 'sep' : ','},
    'diabetes': {'name': 'diabetes.csv', 'to_drop' : [], 'target': 'class', 'sep' : ','},
    'heart': {'name': 'heart.csv', 'to_drop' : [], 'target': 'target', 'sep' : ','},
    'prostate': {'name': 'prostate_cancer.csv', 'to_drop' : [], 'target': 'diagnosis_result', 'sep': ','},
    'breast': {'name': 'breast_cancer.csv', 'to_drop' : ['id'], 'target': 'diagnosis', 'sep': ';'},
    'orthopedic_single': {'name': 'orthopedic_single.csv', 'to_drop' : [], 'target': 'class', 'sep': ','},
    'orthopedic_multi': {'name': 'orthopedic_multi.csv', 'to_drop' : [], 'target': 'class', 'sep': ','},
}


def k_fold_cross_validation(X_train, y_train, knn, dataset_name, n_folds):

    ## K-fold cross validation

    k_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    k_fold_cross_validator = StratifiedKFold(n_splits=n_folds)
    epoch = 0

    for k in range(1, 10):

        accuracy_sum = 0
        precision_sum = 0
        f1_score_sum = 0
        recall_sum = 0

        for train_index, val_index in k_fold_cross_validator.split(X_train, y_train):
            x_train_fold, x_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            prediction = knn.fit_knn_model(k, x_train_fold, x_val_fold, y_train_fold)  # y_train is used to know the class to vote

            ## Compute metrics
            accuracy_sum += accuracy_score(y_val_fold, prediction)
            precision_sum += precision_score(y_val_fold, prediction, average='macro')
            recall_sum += recall_score(y_val_fold, prediction, average = 'macro')
            f1_score_sum += f1_score(y_val_fold, prediction, average='macro')

        ## Epoch completed, hyperparameters evaluated, compute average values over the folds
        k_list.append(k)
        accuracy_list.append(round(accuracy_sum / n_folds, 5))
        precision_list.append(round(precision_sum / n_folds, 5))
        recall_list.append(round((recall_sum / n_folds), 5))
        f1_score_list.append(round(f1_score_sum / n_folds, 5))

        print("Epoch " + str(epoch) + " finished")
        epoch += 1

    ##Save final results on CSV file

    df = pd.DataFrame({'K': k_list, 'Accuracy': accuracy_list, 'Precision': precision_list, 'Recall': recall_list,'F1 score': f1_score_list})
    df.to_csv("tuning/tuning_" + dataset_name + "_" + knn.distance_metric.get_distance_name() + ".csv", index=False, float_format = '%.5f')


def final_evaluation_on_test(k, X_train, X_test, y_train, y_test, knn, dataset_name):

    prediction = knn.fit_knn_model(k, X_train, X_test, y_train)  # y_train is used to know the class to vote

    ## Compute metrics
    accuracy_metric = round(accuracy_score(y_test, prediction), 5)
    precision_metric = round(precision_score(y_test, prediction, average='macro'), 5)
    recall_metric = round(recall_score(y_test, prediction, average='macro'))
    f1_score_metric = round(f1_score(y_test, prediction, average='macro'))

    ##Save final results on CSV file
    df = pd.DataFrame({'K': k, 'Accuracy': accuracy_metric, 'Precision': precision_metric, 'Recall': recall_metric, 'F1 score': f1_score_metric}, index = [1])
    df.to_csv("tuning/test_" + dataset_name + "_" + knn.distance_metric.get_distance_name() + ".csv", index = False)


def main():

    ## Computation parameters

    tuning = False
    n_folds = 3

    dataset_name = 'orthopedic_multi'
    distance_metric = EuclideanDistance()
    #distance_metric.set_cosine_weight(0.7)

    ## Main

    knn = Knn(experiment_dict, dataset_name, distance_metric)
    X, y = prepare_dataset(experiment_dict, dataset_name)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    if tuning:
        k_fold_cross_validation(X_train, y_train, knn, dataset_name, n_folds)
    else:
        final_evaluation_on_test(9, X_train, X_test, y_train, y_test, knn, dataset_name)


if __name__ == "__main__":
    main()


