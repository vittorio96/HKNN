from sklearn.metrics import accuracy_score
from sklearn.model_selection import  KFold

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
    k = 3
    dataset_name = 'ucb_1'
    distance_metric = EuclideanDistance()

    ## Main
    knn = Knn(experiment_dict, dataset_name, distance_metric)
    X, Y = prepare_dataset(experiment_dict, dataset_name)
    x_train, x_val, y_train, y_val = split_dataset(X, Y)
    prediction = knn.fit_knn_model(k, x_train, x_val, y_train)#y_train is used to know the class to vote

    ## Print the accuracy score
    print('Accuracy:', accuracy_score(y_val, prediction))

    k_fold_cross_validator = KFold(n_splits=10)

    for train_index, test_index in k_fold_cross_validator.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        prediction = knn.fit_knn_model(k, x_train, x_val, y_train)  # y_train is used to know the class to vote
        ## Print the accuracy score
        print('Accuracy:', accuracy_score(y_val, prediction))




if __name__ == "__main__":
    main()


