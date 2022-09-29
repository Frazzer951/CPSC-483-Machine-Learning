from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.preprocessing import MinMaxScaler

from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

# variables
order = 2  # order of data
lam = 10**-4  # Reduction tuning parameter
iters = 20_000  # number of iterations
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")


split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
data_X = MinMaxScaler().fit_transform(data_X)
print("Order set")


print(data_X)
print()

# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]


# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]


# Create linear regression objects
r_regr = Ridge(alpha=lam, solver="sag", max_iter=iters)
l_regr = linear_model.Lasso(alpha=lam, max_iter=iters)
e_regr = ElasticNet(alpha=lam, max_iter=iters)


# Train the models using the training sets
print("Start Training")
start_time = time()
r_regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")
print()

print("Start Training")
start_time = time()
l_regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")
print()

print("Start Training")
start_time = time()
e_regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")
print()


# Ridge reduction of model

print(f"Running Ridge Reduction On The Model")
# Make predictions using the testing set
data_y_pred_test = r_regr.predict(data_X_test)
data_y_pred_train = r_regr.predict(data_X_train)
print("Coefficients:")
print(r_regr.coef_)

# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")
print()


# LASSO reduction of model

print(f"Running LASSO Reduction On The Model")
# Make predictions using the testing set
data_y_pred_test = l_regr.predict(data_X_test)
data_y_pred_train = l_regr.predict(data_X_train)
print("Coefficients:")
print(l_regr.coef_)

# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")
print()


# Elastic reduction of model

print(f"Running Elastic Reduction On The Model")
# Make predictions using the testing set
data_y_pred_test = e_regr.predict(data_X_test)
data_y_pred_train = e_regr.predict(data_X_train)
print("Coefficients:")
print(e_regr.coef_)

# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_train, data_y_pred_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_train, data_y_pred_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_test, data_y_pred_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_test, data_y_pred_test):.2f}")
print()
