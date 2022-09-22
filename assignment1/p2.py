from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order, predict

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:

Order 2:

Order 3:

Order 4:

"""

"""
#1
b)
c)
d)
e)
f)
"""

# Load the dataset
df = pd.read_csv("Data1.csv")

percent = 0.5  # Amount of data used for training
split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy()
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=1)
print("Order set")


# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]


# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]


# Train the model using the training sets
print("Start Training")
start_time = time()

# Gradient Descent
n, k = data_X_train.shape
w = np.array([1.0 for _ in range(k + 1)])
alpha = 0.0004  # tuning parameter
for iteration in range(10):
    print(f"Iteration {iteration}")
    print(f"Weight {w}")
    new_w = np.zeros((5,))
    for j in range(k + 1):
        running_sum = 0
        for i, row in enumerate(data_X_train):
            feats = [1]
            feats.extend(row)
            pred = w.T.dot(feats)
            error = pred - data_y_train[i][0]
            running_sum += error * feats[j]
        dw = running_sum / n
        # print(f"j: {j}, dw: {dw}, w[j]: {w[j]}")
        new_w[j] = w[j] - (alpha * dw)
    w = new_w
print(f"Training Took {time()-start_time:.2} seconds")

print(f"Final Weight {w}")

# Make predictions using the testing set
data_y_pred_test = predict(w, data_X_test)
data_y_pred_train = predict(w, data_X_train)
print("Coefficients:")
print(w)
# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")

# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")
