from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)


# variables
order = 1  # order of data
percent = 0.5  # Amount of data used for training

# Load the dataset
df = pd.read_csv("Data1.csv")
df_tostd = df

split_index = int(len(df) * percent)

data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()
data_X = increase_order(mat=data_X, order=order)
print("Order set")

# data_y_std = df_tostd.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
# data_X_std = df_tostd.drop(columns="Idx").to_numpy()
data_y_std = data_y
data_X_std = StandardScaler().fit_transform(X=data_X, y=data_y_std)

# print(data_X[2])
# print(data_X_std[2])

# Part 3 Modifications Start here


# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]

data_x_std_train = data_X_std[:split_index]
data_x_std_test = data_X_std[split_index:]

# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]

data_y_std_train = data_y_std[:split_index]
data_y_std_test = data_y_std[split_index:]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
print("Start Training")
start_time = time()
regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")

print(data_X)

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

print("Start Training after Standardization")
start_time = time()
regr.fit(data_x_std_train, data_y_std_train)
print(f"Training Took {time()-start_time:.2} seconds")


print("Std Set")
print(data_X_std)
# Make predictions using the testing set
data_y_pred_std_test = regr.predict(data_x_std_test)
data_y_pred_std_train = regr.predict(data_x_std_train)
print("Coefficients:")
print(regr.coef_)
# The mean squared error
print(f"RMSE Train : {sqrt(mse(data_y_std_train, data_y_pred_std_train)).real:.2f}")
# The mean squared error
print(f"R^2 Train: {r2_score(data_y_std_train, data_y_pred_std_train):.2f}")

# The mean squared error
print(f"RMSE Test : {sqrt(mse(data_y_std_test, data_y_pred_std_test)).real:.2f}")
# The mean squared error
print(f"R^2 Test: {r2_score(data_y_std_test, data_y_pred_std_test):.2f}")
