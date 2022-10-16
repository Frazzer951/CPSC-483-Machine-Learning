from cmath import sqrt
from time import time

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

"""
Order 1:
Training Took 0.028 seconds
Coefficients:
[ 4.32e-02 -1.02e-03 -6.78e+01 -5.49e-02]
RMSE Train : 0.12
R^2 Train: 0.77
RMSE Test : 0.41
R^2 Test: -1.63

Order 2:
Training Took 0.042 seconds
Coefficients:
[ 4.71e-02 -2.95e-03  2.63e+02 -2.12e-01  4.16e-06  1.89e-05 -5.72e+03  1.87e-04]
RMSE Train : 0.11
R^2 Train: 0.80
RMSE Test : 0.92
R^2 Test: -12.25

Order 3:
Training Took 0.06 seconds
Coefficients:
[ 1.18e+00 -9.92e-03 -4.42e+03  7.96e-01 -3.96e-03  2.12e-04  1.49e+05 -2.39e-03  4.60e-06 -1.30e-06 -1.70e+06  2.20e-06]
RMSE Train : 0.11
R^2 Train: 0.82
RMSE Test : 10.16
R^2 Test: -1628.58
"""


# variables
order = 3  # order of data
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")

split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
print("Order set")


print(data_X)

# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]


# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]


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
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")
