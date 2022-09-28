from cmath import sqrt
from time import time
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import increase_order, normalize

np.set_printoptions(precision=2, linewidth=127)

# variables
lam = 0.001  # Reduction tuning parameter
alpha = 0.01  # tuning parameter
iters = 2_000  # number of iterations
order = 1  # order of data
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

# Split the data into training/testing sets
data_X_train = data_X[:split_index]
data_X_test = data_X[split_index:]


# Split the targets into training/testing sets
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]

print(f"Least Squares Method")

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
print("Start Training")
start_time = time()

regr.fit(data_X_train, data_y_train)
print(f"Training Took {time()-start_time:.2} seconds")
print()


print(f"Running Ridge Reduction On The Model")
W = regr.coef_.copy()

# Ridge reduction of model
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


print(f"Running LASSO Reduction On The Model")
# LASSO reduction of model
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


print(f"Running Elastic Reduction On The Model")
# Elastic reduction of model (LASSO has already been applied, so we will just reapply Ridge)
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
print()


print(f"Gradient Descent Method")
print()

# Gradient Descent
n, k = data_X_train.shape
w = np.ones(k)

print(f"Running Ridge Reduction On The Model")

# Ridge Reduction of model
data_X_trans = data_X_train.T
for iteration in range(iters):
    pred = np.dot(data_X_train, w)
    loss = pred - data_y_train

    if iteration % 1000 == 0:
        # Print out the cost and weights every 1000 iterations
        cost = np.sum(loss**2) / (2 * n)
        print(f"Iteration {iteration} | Cost {cost:.2f} | Weights: {w}")

    gradient = np.dot(data_X_trans, loss) / n

    w = w * (1 - 2 * lam * alpha) - (alpha * gradient)

print(f"Training Took {time()-start_time:.2f} seconds")

print(f"Final Weight {w}")

# Save Weights
Ridge_Gradient_Weights = w.copy()

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
print()

""" LASSO not working

print(f"Running LASSO Reduction On The Model")

# LASSO Reduction of model
w = np.ones(k)


for iteration in range(iters):
    # pred = np.dot(data_X_train, w)
    # loss = pred - data_y_train

    # if iteration % 1000 == 0:
    #     # Print out the cost and weights every 1000 iterations
    #     cost = np.sum(loss**2) / (2 * n)
    #     print(f"Iteration {iteration} | Cost {cost:.2f} | Weights: {w}")

    # gradient = np.dot(data_X_trans, loss) / n

    # w = w * (1 - 2*lam*alpha) - (alpha * gradient)
    if iteration % 10 == 0:
        # Print out the cost and weights every 1000 iterations
        cost = np.sum(loss**2) / (2 * n)
        print(f"Iteration {iteration} | Cost {cost:.2f} | Weights: {w}")

    pred_rows = []
    for i in range(n):
        pred_rows.append(0)
        for m in range(k):
            pred_rows[i] += data_X_train[i][m] * w[m]

    rho = np.zeros(k)
    for j in range(k):
        for i in range(n):
            error = pred_rows[i] - data_X_train[i][j] * w[j]
            rho[j] += data_X_train[i][j] * error
    
    for j in range(k):
        if rho[j] < -lam/2:
            w[j] = rho[j] + lam/2
        elif rho[j] > lam/2:
            w[j] = rho[j] - lam/2
        else:
            w[j] = 0


print(f"Training Took {time()-start_time:.2f} seconds")

print(f"Final Weight {w}")

# Save Weights

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
print()
"""
