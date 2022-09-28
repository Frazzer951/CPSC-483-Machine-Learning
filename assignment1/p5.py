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
K_folds = 4  # Number of partitions the data will be divided into

# Load the dataset
df = pd.read_csv("Data1.csv")


partition_size = len(df) / K_folds

imported_data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
imported_data_X = df.drop(columns="Idx").to_numpy()


imported_data_X = increase_order(mat=imported_data_X, order=order)
imported_data_X = normalize(imported_data_X)
print("Order set")


print(imported_data_X)
print()

# Split the data into equally sized sets
data_X = []
for k in range(K_folds):
    data_X.append(imported_data_X[int(k * partition_size) : int((k + 1) * partition_size)])


# Split the targets into equally sized sets
data_y = []
for k in range(K_folds):
    data_y.append(imported_data_y[int(k * partition_size) : int((k + 1) * partition_size)])


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
print("Start Training")
start_time = time()


Ridge_models = []
Lasso_models = []
Elastic_models = []

for k in range(K_folds):
    regr.fit(data_X[k], data_y[k])
    print(f"Training Took {time()-start_time:.2} seconds")
    print()

    W = regr.coef_.copy()

    # Ridge reduction of model
    print("Running Ridge Reduction On The Model")

    for i in range(len(regr.coef_)):
        regr.coef_[i] = regr.coef_[i] + lam * regr.coef_[i] ** 2

    # Add model to list of Ridge models
    Ridge_models.append(regr.coef_)

    # Make predictions using the testing set
    data_y_pred_test = regr.predict(data_X[k])
    data_y_pred_train = regr.predict(data_X[k])
    print("Coefficients:")
    print(regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y[k], data_y_pred_train):.2f}")

    # The mean squared error
    print(f"RMSE Test : {sqrt(mse(data_y[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    print(f"R^2 Test: {r2_score(data_y[k], data_y_pred_test):.2f}")
    print()

    # LASSO reduction of model
    print("Running LASSO Reduction On The Model")

    regr.coef_ = W.copy()
    for i in range(len(regr.coef_)):
        regr.coef_[i] = regr.coef_[i] + lam * abs(regr.coef_[i])

    # Add model to list of LASSO models
    Lasso_models.append(regr.coef_)

    # Make predictions using the testing set
    data_y_pred_test = regr.predict(data_X[k])
    data_y_pred_train = regr.predict(data_X[k])
    print("Coefficients:")
    print(regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y[k], data_y_pred_train):.2f}")

    # The mean squared error
    print(f"RMSE Test : {sqrt(mse(data_y[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    print(f"R^2 Test: {r2_score(data_y[k], data_y_pred_test):.2f}")
    print()

    # Elastic reduction of model (LASSO has already been applied, so we will just reapply Ridge)
    print("Running Elastic Reduction On The Model")

    for i in range(len(regr.coef_)):
        regr.coef_[i] = regr.coef_[i] + lam * regr.coef_[i] ** 2

    # Add model to list of Elastic models
    Elastic_models.append(regr.coef_)

    # Make predictions using the testing set
    data_y_pred_test = regr.predict(data_X[k])
    data_y_pred_train = regr.predict(data_X[k])
    print("Coefficients:")
    print(regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y[k], data_y_pred_train):.2f}")

    # The mean squared error
    print(f"RMSE Test : {sqrt(mse(data_y[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    print(f"R^2 Test: {r2_score(data_y[k], data_y_pred_test):.2f}")
    print()

    regr.coef_ = W.copy()
# End of for loop

# Get the mean weights for Ridge model
Sums = []
for j in range(len(Ridge_models[0])):
    Sums.append(0)
    for i in range(len(Ridge_models)):
        Sums[j] += Ridge_models[i][j]
for sum in Sums:
    sum /= len(Ridge_models)

# Mean Ridge model
print("Mean Weights Of Ridge Models:")
print(Sums)
print()

# Get the mean weights for LASSO model
Sums = []
for j in range(len(Lasso_models[0])):
    Sums.append(0)
    for i in range(len(Lasso_models)):
        Sums[j] += Lasso_models[i][j]
for sum in Sums:
    sum /= len(Lasso_models)

# Mean LASSO model
print("Mean Weights Of LASSO Models:")
print(Sums)
print()

# Get the mean weights for Elastic model
Sums = []
for j in range(len(Elastic_models[0])):
    Sums.append(0)
    for i in range(len(Elastic_models)):
        Sums[j] += Elastic_models[i][j]
for sum in Sums:
    sum /= len(Elastic_models)

# Mean Ridge model
print("Mean Weights Of Elastic Models:")
print(Sums)
print()
