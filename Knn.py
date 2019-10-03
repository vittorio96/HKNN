import numpy as np
from distances.EuclideanDistance import EuclideanDistance
from distances.Distance import Distance
from utils.preprocessor import  prepare_dataset
from utils.dataset_splitter import split_dataset

from sklearn.metrics import  accuracy_score

class Knn:

    def __init__(self, datasets, dataset, distance_metric):
        self.datasets = datasets
        self.dataset = dataset
        self.distance_metric = distance_metric

    def sort_similarities_bubble_sort(self, similarities, targets):
        """
            :type similarities: numpy.ndarray, targets: numpy.ndarray
            :rtype: similarites: numpy.ndarray, targets: numpy.ndarray
        """
        temp_target = targets.tolist()
        for i in range(len(similarities)):
            for j in range(0, len(similarities) - i - 1):
                if (similarities[j + 1] < similarities[j]):
                    similarities[j], similarities[j + 1] = similarities[j + 1], similarities[j]
                    ## Sort the classes along with the eucledian distances to maintain relevancy
                    temp_target[j], temp_target[j + 1] = temp_target[j + 1], temp_target[j]

        return temp_target, similarities


    def fit_knn_model_original(self, k):

        X, Y = prepare_dataset(self.datasets, self.dataset)

        x_train, x_val, y_train, y_val = split_dataset(X, Y)

        distance_matrix = np.ndarray

        ## Run the KNN classifier

        y_pred_knn = []


        ## Iterate through each value in test data
        for val in x_val:

            euc_dis = []

            ## Finding eucledian distance for all points in training data
            for point in x_train:
                euc_dis.append(self.distance_metric.compute_distance(val, point))

            ##Combine similarites
            average_distance = euc_dis

            ## A temporary target array is created and both similarities and targets are sorted in a rank with bubble-sort
            temp_target = y_train
            temp_target, euc_dis = self.sort_similarities_bubble_sort(average_distance, temp_target)

            print(euc_dis)
            print("---")

            ## Finding majority among the neighbours for the selected point
            vote = [0 for _ in range(k)]
            for i in range(k):
                vote[temp_target[i]] += 1 #use the class as hashing solution

            y_pred_knn.append(vote.index(max(vote)))#final prediction for the selected point

        ## Print the accuracy score
        print('Accuracy:',accuracy_score(y_val,y_pred_knn))

    def fit_knn_model(self, dataset, k):

        X, Y = prepare_dataset(self.datasets, self.dataset)

        x_train, x_val, y_train, y_val = split_dataset(X, Y)

        ## Run the KNN classifier

        y_pred_knn = []

        distance = EuclideanDistance()

        ## Iterate through each value in test data
        for val in x_val:

            euc_dis = []

            ## Finding eucledian distance for all points in training data
            for point in x_train:
                euc_dis.append(distance.compute_distance(val, point))

            ##Combine similarites
            average_distance = euc_dis

            ## A temporary target array is created and both similarities and targets are sorted in a rank with bubble-sort
            temp_target = y_train
            temp_target, euc_dis = self.sort_similarities_bubble_sort(average_distance, temp_target)

            ## Finding majority among the neighbours for the selected point
            vote = [0 for _ in range(k)]
            for i in range(k):
                vote[temp_target[i]] += 1 #use the class as hashing solution

            y_pred_knn.append(vote.index(max(vote)))#final prediction for the selected point

        ## Print the accuracy score
        print('Accuracy:',accuracy_score(y_val,y_pred_knn))