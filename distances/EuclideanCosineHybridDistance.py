from distances.Distance import Distance
from distances.EuclideanDistance import EuclideanDistance
from distances.CosineDistance import CosineDistance
from sklearn.preprocessing import scale


class EuclideanCosineHybridDistance(Distance):

    def set_hybrid_weight(self, alpha):
        self.alpha = alpha

    def compute_distances_from_points(self, point1, points_list):

        euclidean_distances,cosine_distances = [], []
        euclidean, cosine = EuclideanDistance(), CosineDistance()

        ## Finding eucledian distance for all points in training data
        for point in points_list:
            euclidean_distances.append(euclidean.compute_distance(point1, point))
            cosine_distances.append(cosine.compute_distance(point1, point))

        ## Normalize the distances and create the weighted sum
        return (self.alpha * scale(euclidean_distances) + (1-self.alpha)*scale(cosine_distances))


    def get_distance_name(self):
        return "EuclideanCosineHybrid"