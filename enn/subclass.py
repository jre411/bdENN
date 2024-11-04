"""The Subclass object"""
import numpy as np

class Subclass():
    """An ENN subclass, containing its class and points and other attributes"""
    def __init__(self, y_class=0, points=None, level=0):
        self.y_class = y_class #The superordinate class
        if points is None:
            points = []
        self.points = points #The points that belong to the subclass
        self.support_vectors = np.zeros(len(points), dtype=bool) #Whether each point serves as a support vector
        self.first_margin = None #The SVM margin found before differentia pruning
        self.first_misclass = None #The misclassification error before differentia pruning
        self.level = level

    def reset_points(self, points):
        """Resets the points and support_vector assignments"""
        self.points = points
        self.support_vectors = np.zeros(len(points), dtype=bool)
