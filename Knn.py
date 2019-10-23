from distances.EuclideanDistance import EuclideanDistance
import numpy as np

class Knn:

    def __init__(self, experiment_dict, dataset, distance_metric):
        self.datasets = experiment_dict
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


    def fit_knn_model(self, k, x_train, x_val, y_train):

        ## Run the KNN classifier
        y_pred_knn = []

        distance = EuclideanDistance()

        ## Iterate through each value in test data
        for val in x_val:

            distances = []

            ## Finding eucledian distance for all points in training data
            for point in x_train:
                distances.append(distance.compute_distance(val, point))

            ## A temporary target array is created and both similarities and targets are sorted in a rank with bubble-sort
            temp_target = y_train
            temp_target, distances = self.sort_similarities_bubble_sort(distances, temp_target)

            ## Finding majority among the neighbours for the selected point
            vote = [0 for _ in range(max(temp_target) + 1)]

            for i in range(k):
                vote[temp_target[i]] += 1 #use the class as hashing solution

            y_pred_knn.append(vote.index(max(vote)))#final prediction for the selected point

        return np.array(y_pred_knn)