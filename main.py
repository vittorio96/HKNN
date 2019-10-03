from Knn import Knn
from distances.EuclideanDistance import EuclideanDistance

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
    knn.fit_knn_model_original(k)


if __name__ == "__main__":
    main()


