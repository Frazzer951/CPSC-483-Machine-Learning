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
Training Took 0.025 seconds
Coefficients:
[ 15.25  -0.31  -2.87 -25.58]
RMSE Train : 0.12
R^2 Train: 0.77
RMSE Test : 0.41
R^2 Test: -1.63

Order 2:
Training Took 0.042 seconds
Coefficients:
[ 16.64  -0.89  11.14 -98.53   0.52   1.7  -10.27  40.59]
RMSE Train : 0.11
R^2 Train: 0.80
RMSE Test : 0.92
R^2 Test: -12.25

Order 3:
Training Took 0.069 seconds
Coefficients:
[ 417.13   -2.97 -187.24  370.69 -493.77   19.1   267.8  -519.28  202.78  -35.01 -129.61  222.1 ]
RMSE Train : 0.11
R^2 Train: 0.82
RMSE Test : 10.16
R^2 Test: -1628.57

Order 4:
Training Took 0.084 seconds
Coefficients:
[ 2.65e+03 -1.11e+10 -1.92e+03  1.81e+04 -4.51e+03  1.33e+11  3.68e+03 -3.02e+04  3.39e+03 -6.22e+11 -3.09e+03  2.23e+04
 -9.40e+02  9.44e+11  9.54e+02 -6.16e+03]
RMSE Train : 0.10
R^2 Train: 0.83
RMSE Test : 239292488038.93
R^2 Test: -903425261936573960683520.00

Order 5:
Training Took 0.11 seconds
Coefficients:
[-2.99e+04 -1.22e+10  9.13e+03 -3.28e+05  7.31e+04  1.42e+11 -2.55e+04  7.43e+05 -8.87e+04 -6.21e+11  3.53e+04 -8.41e+05
  5.35e+04  7.20e+11 -2.41e+04  4.75e+05 -1.28e+04  4.99e+11  6.51e+03 -1.07e+05]
RMSE Train : 0.10
R^2 Train: 0.83
RMSE Test : 391560956060.76
R^2 Test: -2418983244784230671056896.00
"""

"""
#1
b)  The training and testing data was split 50/50
c)  RMSE Train : 0.12
    R^2 Train: 0.77
    RMSE Test : 0.41
    R^2 Test: -1.63
    Took 0.025 seconds to train
d)  4 terms in total, with a polynomial order of 1
e)  When training with higher orders, while the training data kept a small error,
    the testing data had a huge increase in error. This leads me to believe that
    the model becomes over fit with any order greater than 1
"""

# variables
order = 1  # order of data
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
