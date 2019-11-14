from distances.Distance import Distance
from distances.CosineDistance import CosineDistance
from distances.ManhattanDistance import ManhattanDistance
from sklearn.preprocessing import scale


class CosineManhattanHybridDistance(Distance):

    def set_cosine_weight(self, alpha):
        self.alpha = alpha

    def compute_distances_from_points(self, point1, points_list):
        manhattan_distances, cosine_distances = [], []
        manhattan, cosine = ManhattanDistance(), CosineDistance()

        ## Finding eucledian distance for all points in training data
        for point in points_list:
            manhattan_distances.append(manhattan.compute_distance(point1, point))
            cosine_distances.append(cosine.compute_distance(point1, point))

        ## Normalize the distances and create the weighted sum
        return (self.alpha * scale(manhattan_distances) + scale(cosine_distances))

    def get_distance_name(self):
        return "CosineManhattanHybrid"