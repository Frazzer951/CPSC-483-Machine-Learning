from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order

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
f)s
"""

"""
20k
Training Took 113.24 seconds
Final Weight [ 1.66e-03 -1.33e-01  1.49e-02 -2.64e-05  2.23e-01]
Coefficients:
[ 1.66e-03 -1.33e-01  1.49e-02 -2.64e-05  2.23e-01]
RMSE Train : 1.12
R^2 Train: -18.87
RMSE Test : 0.95
R^2 Test: -13.37

15k
Training Took 81.53 seconds
Final Weight [ 1.53e-03 -1.27e-01  1.67e-02 -2.45e-05  2.18e-01]
Coefficients:
[ 1.53e-03 -1.27e-01  1.67e-02 -2.45e-05  2.18e-01]
RMSE Train : 1.13
R^2 Train: -19.29
RMSE Test : 0.94
R^2 Test: -13.08

10k
Training Took 55.02 seconds
Final Weight [ 1.34e-03 -1.11e-01  2.15e-02 -2.10e-05  2.06e-01]
Coefficients:
[ 1.34e-03 -1.11e-01  2.15e-02 -2.10e-05  2.06e-01]
RMSE Train : 1.21
R^2 Train: -22.24
RMSE Test : 1.54
R^2 Test: -36.54

"""

# variables
order = 1
alpha = 0.000001  # tuning parameter
iters = 15_000

# Load the dataset
df = pd.read_csv("Data1.csv")

percent = 0.5  # Amount of data used for training
split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
print("Order set")

n, _ = data_X.shape
ones = np.ones((n, 1))
data_X = np.hstack((ones, data_X))


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
        cst = np.sum(loss**2) / (2 * n)
        print(f"Iteration {iteration} | Cost {cst:.2f} | Weights: {w}")

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
