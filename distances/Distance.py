from abc import ABC, abstractmethod

"""
@author: Vittorio Maria Enrico Denti, Cornelis Tim Brinkman
"""

class Distance(ABC):

    @abstractmethod
    def compute_distance(self, point1, point2):
        pass