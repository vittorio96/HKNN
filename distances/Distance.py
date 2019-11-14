from abc import ABC, abstractmethod

"""
@author: Vittorio Maria Enrico Denti, Cornelis Tim Brinkman
"""

class Distance(ABC):

    @abstractmethod
    def compute_distances_from_points(self, point1, points_list):
        pass

    @abstractmethod
    def get_distance_name(self):
        pass