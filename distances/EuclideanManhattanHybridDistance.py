from distances.Distance import Distance
from distances.EuclideanDistance import EuclideanDistance
from distances.ManhattanDistance import ManhattanDistance
from sklearn.preprocessing import scale


class EuclideanManhattanHybridDistance(Distance):

    def set_euclidean_weight(self, alpha):
        self.alpha = alpha

    def compute_distances_from_points(self, point1, points_list):
        euclidean_distances, manhattan_distances = [], []
        euclidean, manhattan = EuclideanDistance(), ManhattanDistance()

        ## Finding eucledian distance for all points in training data
        for point in points_list:
            euclidean_distances.append(euclidean.compute_distance(point1, point))
            manhattan_distances.append(manhattan.compute_distance(point1, point))

        ## Normalize the distances and create the weighted sum
        return (self.alpha * scale(euclidean_distances) + scale(manhattan_distances))

    def get_distance_name(self):
        return "EuclideanManhattanHybrid"