from distances.Distance import Distance


class EuclideanDistance(Distance):

    ## Ovverride
    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        if len(point1) != len(point2):
            return -1

        cumulative_distance = 0
        for i in range(0, len(point1)):
            cumulative_distance += (point1[i] - point2[i]) ** 2

        return float(cumulative_distance ** 0.5)
