from sklearn.metrics import accuracy_score
from sklearn.model_selection import  StratifiedKFold

from Knn import Knn
from distances.EuclideanDistance import EuclideanDistance
from utils.preprocessor import prepare_dataset
from utils.dataset_splitter import split_dataset

experiment_dict = {
    'ucb_1': {'name': 'Iris.csv', 'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], 'target': 'Species'},
    'ucb_2': 'name2.csv', 'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], 'target': 'Species',
    'ucb_3': 'name3.csv', 'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    'ucb_4': 'name4.csv', 'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    'ucb_5': 'name5.csv', 'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']}

def main():

    ## Initial parameters
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    dataset_name = 'ucb_1'
    distance_metric = EuclideanDistance()

    ## Main
    knn = Knn(experiment_dict, dataset_name, distance_metric)
    X, Y = prepare_dataset(experiment_dict, dataset_name)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    ## K-fold cross validation

    k_fold_cross_validator = StratifiedKFold(n_splits=10)

    for train_index, val_index in k_fold_cross_validator.split(X_train, Y_train):
        for k in k_list:
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]
            prediction = knn.fit_knn_model(k, x_train, x_val, y_train)  # y_train is used to know the class to vote
            ## Print the accuracy score
            print('Accuracy:', accuracy_score(y_val, prediction))




if __name__ == "__main__":
    main()


