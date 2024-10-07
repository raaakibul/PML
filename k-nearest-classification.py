# Supervised algorithm
# weight/height 
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
# print(data)
print(data.feature_names)
print(data.target_names)