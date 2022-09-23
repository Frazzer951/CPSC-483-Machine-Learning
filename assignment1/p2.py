from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order, normalize

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:
Training Took 109.49 seconds
Final Weight [37.32  1.42  5.53 -2.51 18.48]
Coefficients:
[37.32  1.42  5.53 -2.51 18.48]
RMSE Train : 0.83
R^2 Train: -9.93
RMSE Test : 2.76
R^2 Test: -119.59

Order 2:
Training Took 178.53 seconds
Final Weight [30.49  8.35  3.76  5.51 18.25 -7.47  1.48 -9.83  8.26]
Coefficients:
[30.49  8.35  3.76  5.51 18.25 -7.47  1.48 -9.83  8.26]
RMSE Train : 0.57
R^2 Train: -4.09
RMSE Test : 1.68
R^2 Test: -43.65

Order 3:
Training Took 249.28 seconds
Final Weight [25.04  9.96  3.15  8.02 16.15 -0.44  1.31 -2.14  8.97 -7.5   1.08 -7.93  3.19]
Coefficients:
[25.04  9.96  3.15  8.02 16.15 -0.44  1.31 -2.14  8.97 -7.5   1.08 -7.93  3.19]
RMSE Train : 0.70
R^2 Train: -6.71
RMSE Test : 1.65
R^2 Test: -41.73

Order 4:
Training Took 337.38 seconds
Final Weight [22.51 10.11  2.76  8.62 14.84  1.96  1.18  0.73  8.74 -3.21  1.03 -3.46  3.9  -6.31  1.01 -5.41  0.11]
Coefficients:
[22.51 10.11  2.76  8.62 14.84  1.96  1.18  0.73  8.74 -3.21  1.03 -3.46  3.9  -6.31  1.01 -5.41  0.11]
RMSE Train : 0.81
R^2 Train: -9.44
RMSE Test : 1.99
R^2 Test: -61.18

Order 5:
Training Took 415.44 seconds
Final Weight [21.51 10.02  2.34  8.74 14.26  2.79  1.02  1.83  8.54 -1.5   0.97 -1.55  4.07 -3.77  0.99 -2.85  0.63 -4.69  1.   -2.98 -1.99]
Coefficients:
[21.51 10.02  2.34  8.74 14.26  2.79  1.02  1.83  8.54 -1.5   0.97 -1.55  4.07 -3.77  0.99 -2.85  0.63 -4.69  1.   -2.98 -1.99]
RMSE Train : 0.87
R^2 Train: -10.97
RMSE Test : 2.49
R^2 Test: -96.94
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
alpha = 0.01  # tuning parameter
iters = 20_000  # number of iterations
order = 5  # order of data
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")

split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
data_X = normalize(data_X)
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
