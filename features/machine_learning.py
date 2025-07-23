import pandas as pd
import numpy as np

class MinMaxScaler:

    def __init__(self):

        self.min = np.array([3.0, 16.0, 11.0, 1.0, 4.0, 0.0])
        self.max = np.array([163.0, 287.0, 89.0, 55.0, 81.0, 53.0])

    def transform(self, x):
        
        x_scaled = (x - self.min) / (self.max - self.min)
        x_scaled = x_scaled * (1 - 0) + 0

        return x_scaled

    def inverse_transform(self, x_scaled):
        
        x_original = (x_scaled - 0) / (1 - 0)
        return x_original * (self.max - self.min) + self.min
    
class ManhattanKNN:

    def __init__(self):
        
        self.x_train = np.load('features/x.npy')
        self.y_train = np.load('features/y.npy')

        self.n = 2

    def distance_metric(self, x):
        return np.sum(np.abs(self.x_train - x), axis=1)

    def predict(self, x_test):
        
        result = []
        x_test = np.array(x_test)

        for x in x_test:
            class_dict = {}

            distance = self.distance_metric(x)
            nearest_neighbors = np.argsort(distance)[:self.n]
            classes = self.y_train[nearest_neighbors].astype(int)

            for cls in classes:
                class_dict[cls] = class_dict.get(cls, 0) + 1

            result.append(max(class_dict, key=class_dict.get))

        return result