from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order, normalize

np.set_printoptions(precision=2, linewidth=127)

# variables
order = 2  # order of data
lam = 10**-5  # Reduction tuning parameter
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")


split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


data_X = increase_order(mat=data_X, order=order)
data_X = normalize(data_X)
print("Order set")


print(data_X)
print()

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
print()

W = regr.coef_.copy()

# Ridge reduction of model
print(f"Running Ridge Reduction On The Model")

for i in range(len(regr.coef_)):
    regr.coef_[i] = regr.coef_[i] + lam * regr.coef_[i] ** 2


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
print()


# LASSO reduction of model
print(f"Running LASSO Reduction On The Model")

regr.coef_ = W.copy()
for i in range(len(regr.coef_)):
    regr.coef_[i] = regr.coef_[i] + lam * abs(regr.coef_[i])


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
print()


# Elastic reduction of model (LASSO has already been applied, so we will just reapply Ridge)
print(f"Running Elastic Reduction On The Model")

for i in range(len(regr.coef_)):
    regr.coef_[i] = regr.coef_[i] + lam * regr.coef_[i] ** 2


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
print()
