from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:
Training Took 0.023 seconds
Coefficients:
[[ 4.32e-02 -1.02e-03 -6.78e+01 -5.49e-02]]
RMSE Test : 0.41
R^2 Test: -1.63
RMSE Train : 0.12
R^2 Train: 0.77

Order 2:
Training Took 0.041 seconds
Coefficients:
[[ 4.71e-02 -2.95e-03  2.63e+02 -2.12e-01  4.16e-06  1.89e-05 -5.72e+03  1.87e-04]]
RMSE Test : 0.92
R^2 Test: -12.25
RMSE Train : 0.11
R^2 Train: 0.80

Order 3:
Training Took 0.066 seconds
Coefficients:
[[ 1.18e+00 -9.92e-03 -4.42e+03  7.96e-01 -3.96e-03  2.12e-04  1.49e+05 -2.39e-03  4.60e-06 -1.30e-06 -1.70e+06  2.20e-06]]
Root Mean squared error: 10.16
R^2: -1628.58

Order 4:
Training Took 0.086 seconds
Coefficients:
[[ 1.62e+00  2.91e+00  5.54e+02  2.54e+01 -7.39e-03 -1.16e-01 -1.07e+04 -9.21e-02  1.52e-05  1.81e-03 -1.01e+03  1.47e-04
  -1.16e-08 -9.18e-06 -6.35e+01 -8.79e-08]]
Root Mean squared error: 18851.68
R^2: -5607047729.94
"""

# Load the dataset
df = pd.read_csv("Data1.csv")

percent = 0.5  # Amount of data used for training
split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy()
data_X = df.drop(columns="Idx").to_numpy()

# print("data_X:", data_X)

data_X = increase_order(mat=data_X, order=2)
print("Order set")

# print("data_X:", data_X)
# print(type(data_X))
# print("data_y:", data_y)
# print(type(data_y))

# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]
# print("Train:", data_X_train)
# print(len(data_X_train))
# print("Test:", data_X_test)
# print(len(data_X_test))


# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]
# print("Train:", data_y_train)
# print(len(data_y_test))
# print("Test:", data_y_train)
# print(len(data_y_test))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
print("Start Training")
start_time = time()
regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")

# Make predictions using the testing set
data_y_pred_test = regr.predict(data_X_test)
data_y_pred_train = regr.predict(data_X_train)
print("Coefficients:")
print(regr.coef_)
# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")

# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")
