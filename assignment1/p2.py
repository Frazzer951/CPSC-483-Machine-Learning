from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:
Training Took 113.70 seconds
Final Weight [ 1.   -0.12  0.02  1.    0.21]
Coefficients:
[ 1.   -0.12  0.02  1.    0.21]
RMSE Train : 1.12
R^2 Train: -18.77
RMSE Test : 0.94
R^2 Test: -12.98

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


# variables
order = 1
alpha = 0.000001  # tuning parameter
iters = 20_000

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
