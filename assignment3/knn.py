import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, preprocess, split_data

print("Loading Data")
data = load_data()
# print(data)

print("Preprocessing Data")
data = preprocess(data)
# print(data)

print("Spliting Data")
training_data, test_data, training_labels, test_labels = split_data(data)

# print("Training Data")
# print(training_data)
# print("Training Labels")
# print(training_labels)

# print("Test Data")
# print(test_data)
# print("Test Labels")
# print(test_labels)
