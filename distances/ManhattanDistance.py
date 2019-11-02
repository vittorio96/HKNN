from distances.Distance import Distance
from scipy.spatial import minkowski_distance
from scipy.stats import chisquare
from scipy.spatial.distance import cdist


class ManhattanDistance(Distance):

    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        return minkowski_distance(point1, point2, p = 1) ##A Manhattan distance is a Minkowski distance with p = 1

    def get_distance_name(self):
        return "Manhattan"