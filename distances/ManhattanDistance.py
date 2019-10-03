from distances.Distance import Distance


class ManhattanDistance(Distance):

    def compute_distance(self, point1, point2):
        """
            :type point1: numpy.ndarray, point2: numpy.ndarray
            :rtype: cumulative_distance: float, -1 if arrays are not matching
        """
        return 0