from distances.Distance import Distance
from distances.EuclideanDistance import EuclideanDistance
from distances.ManhattanDistance import ManhattanDistance


class EuclideanManhattanHybridDistance(Distance):

    def set_euclidean_weight(self, alpha):
        self.alpha = alpha

    ## Ovverride
    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        euclidean = EuclideanDistance()
        manhattan = ManhattanDistance()

        return self.alpha * euclidean.compute_distance(point1, point2) + (1-self.alpha) * manhattan.compute_distance(point1, point2)

    def get_distance_name(self):
        return "EuclideanManhattanHybrid"