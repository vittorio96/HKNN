from distances.Distance import Distance
from distances.CosineDistance import CosineDistance
from distances.ManhattanDistance import ManhattanDistance


class CosineManhattanHybridDistance(Distance):

    def set_cosine_weight(self, alpha):
        self.alpha = alpha

    ## Ovverride
    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        cosine = CosineDistance()
        manhattan = ManhattanDistance()

        return self.alpha * cosine.compute_distance(point1, point2) + (1-self.alpha) * manhattan.compute_distance(point1, point2)

    def get_distance_name(self):
        return "CosineManhattanHybrid"