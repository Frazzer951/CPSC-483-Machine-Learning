from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.preprocessing import MinMaxScaler

from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:
Training Took 108.48 seconds
Final Weight [52.05 -3.6   2.53 -0.92  7.03]
Coefficients:
[52.05 -3.6   2.53 -0.92  7.03]
RMSE Train : 0.51
R^2 Train: -3.12
RMSE Test : 0.71
R^2 Test: -7.03

Order 2:
Training Took 175.58 seconds
Final Weight [ 5.12e+01 -2.49e+00  2.62e+00 -1.76e-02  5.84e+00 -2.99e+00  1.27e+00 -1.15e+00  4.45e+00]
Coefficients:
[ 5.12e+01 -2.49e+00  2.62e+00 -1.76e-02  5.84e+00 -2.99e+00  1.27e+00 -1.15e+00  4.45e+00]
RMSE Train : 0.64
R^2 Train: -5.49
RMSE Test : 1.05
R^2 Test: -16.30

Order 3:
Training Took 250.78 seconds
Final Weight [50.98 -1.97  2.67  0.53  4.97 -2.11  1.28 -0.33  3.67 -2.2   1.03 -1.    2.44]
Coefficients:
[50.98 -1.97  2.67  0.53  4.97 -2.11  1.28 -0.33  3.67 -2.2   1.03 -1.    2.44]
RMSE Train : 0.69
R^2 Train: -6.55
RMSE Test : 1.56
R^2 Test: -37.33
"""


# variables
alpha = 0.01  # tuning parameter
iters = 20_000  # number of iterations
order = 3  # order of data
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")

split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
data_X = MinMaxScaler().fit_transform(data_X)
print("Order set")

n, _ = data_X.shape
ones = np.ones((n, 1))
data_X = np.hstack((ones, data_X))

print(data_X)

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
w = np.ones(k)

data_X_trans = data_X_train.T
for iteration in range(iters):
    pred = np.dot(data_X_train, w)
    loss = pred - data_y_train

    if iteration % 1000 == 0:
        # Print out the cost and weights every 1000 iterations
        cost = np.sum(loss**2) / (2 * n)
        print(f"Iteration {iteration} | Cost {cost:.2f} | Weights: {w}")

    gradient = np.dot(data_X_trans, loss) / n

    w = w - (alpha * gradient)

print(f"Training Took {time()-start_time:.2f} seconds")

print(f"Final Weight {w}")

# Make predictions using the testing set
data_y_pred_train = np.dot(data_X_train, w).T
data_y_pred_test = np.dot(data_X_test, w).T

print("Coefficients:")
print(w)
# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")
