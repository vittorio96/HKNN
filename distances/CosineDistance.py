from distances.Distance import Distance
from scipy.spatial import distance

class CosineDistance(Distance):

    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        return distance.cosine(point1, point2)

    def compute_distances_from_points(self, point1, points_list):

        distances = []
        ## Finding eucledian distance for all points in training data
        for point in points_list:
            distances.append(self.compute_distance(point1, point))
        return distances

    def get_distance_name(self):
        return "Cosine"