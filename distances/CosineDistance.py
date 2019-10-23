from distances.Distance import Distance
from scipy.spatial import distance

class CosineDistance(Distance):

    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        return distance.cosine(point1, point2)

    def get_distance_name(self):
        return "Cosine"