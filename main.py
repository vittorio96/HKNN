from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import  StratifiedKFold
import pandas as pd

from Knn import Knn
from distances.EuclideanDistance import EuclideanDistance
from distances.ManhattanDistance import ManhattanDistance
from utils.preprocessor import prepare_dataset
from utils.dataset_splitter import split_dataset

experiment_dict = {
    'ucb_iris': {'name': 'Iris.csv', 'to_drop' : [], 'target': 'Species'},
    'ucb_diabetes': {'name': 'diabetes.csv', 'to_drop' : [], 'target': 'class'},
    'ucb_parkinson': {'name': 'parkinson.xls',  'to_drop' : [], 'target': 'status'},
    'ucb_epilepsy': {'name': 'epilepsy.csv', 'to_drop' : [], 'target': 'y'},
    'ucb_5': {'name': 'xxx', 'to_drop' : [], 'target': 'Species'}
}

def main():

    ## Initial parameters

    dataset_name = 'ucb_diabetes'
    distance_metric = EuclideanDistance()

    ## Main

    knn = Knn(experiment_dict, dataset_name, distance_metric)
    X, Y = prepare_dataset(experiment_dict, dataset_name)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    ## K-fold cross validation

    k_list = []
    accuracy_list = []
    precision_list = []
    f1_score_list = []
    n_folds = 2

    k_fold_cross_validator = StratifiedKFold(n_splits=n_folds)
    epoch = 0

    for k in range(1, 3):

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
        accuracy_list.append(round(accuracy_sum/n_folds, 5))
        precision_list.append(round(precision_sum/n_folds, 5))
        f1_score_list.append(round(f1_score_sum/n_folds, 5))

        print("Epoch "+str(epoch)+" finished")
        epoch += 1

    ##Save final results on CSV file

    df = pd.DataFrame({'K': k_list, 'Accuracy': accuracy_list, 'Precision': precision_list, 'F1 score': f1_score_list})
    df.to_csv("tuning/tuning_"+dataset_name+"_"+distance_metric.get_distance_name()+".csv")




if __name__ == "__main__":
    main()


