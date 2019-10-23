from distances.Distance import Distance
from scipy.spatial.distance import cdist


class ManhattanDistance(Distance):

    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        return cdist(point1, point2, metric = "cityblock")

    def get_distance_name(self):
        return "Manhattan"