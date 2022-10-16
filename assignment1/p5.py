from cmath import sqrt
from time import time

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from utils import increase_order

np.set_printoptions(precision=2, linewidth=127)

# variables
order = 1  # order of data
lam = 10**-3  # Reduction tuning parameter
iters = 20_000  # number of iterations
K_folds = 5  # Number of partitions the data will be divided into

# Load the dataset
df = pd.read_csv("Data1.csv")


partition_size = len(df) / K_folds

imported_data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
imported_data_X = df.drop(columns="Idx").to_numpy()


imported_data_X = increase_order(mat=imported_data_X, order=order)
print("Order set")


print(imported_data_X)
print()

# Split the data into 'K' equally sized sets
data_X_train = []
data_X_test = []
for k in range(K_folds):
    data_X_train.append(np.delete(imported_data_X, slice(int(k * partition_size), int((k + 1) * partition_size)), 0))
    data_X_test.append(imported_data_X[int(k * partition_size) : int((k + 1) * partition_size)])


# Split the targets into 'K' equally sized sets
data_y_train = []
data_y_test = []
for k in range(K_folds):
    data_y_train.append(np.delete(imported_data_y, slice(int(k * partition_size), int((k + 1) * partition_size)), 0))
    data_y_test.append(imported_data_y[int(k * partition_size) : int((k + 1) * partition_size)])

Ridge_rmse = []
Ridge_rsqr = []
Lasso_rmse = []
Lasso_rsqr = []
Elastic_rmse = []
Elastic_rsqr = []

for k in range(K_folds):

    # Create linear regression object
    r_regr = Ridge(alpha=lam, solver="sag", max_iter=iters)
    l_regr = linear_model.Lasso(alpha=lam, max_iter=iters)
    e_regr = ElasticNet(alpha=lam, max_iter=iters)

    # Train the models using the training sets
    print("Start Training")
    start_time = time()
    r_regr.fit(data_X_train[k], data_y_train[k])
    print(f"Training Took {time()-start_time:.2} seconds")
    print()

    print("Start Training")
    start_time = time()
    l_regr.fit(data_X_train[k], data_y_train[k])
    print(f"Training Took {time()-start_time:.2} seconds")
    print()

    print("Start Training")
    start_time = time()
    e_regr.fit(data_X_train[k], data_y_train[k])
    print(f"Training Took {time()-start_time:.2} seconds")
    print()

    # Ridge reduction of model

    print(f"Running Ridge Reduction On The Model")
    # Make predictions using the testing set
    data_y_pred_test = r_regr.predict(data_X_test[k])
    data_y_pred_train = r_regr.predict(data_X_train[k])
    print("Coefficients:")
    print(r_regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y_train[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y_train[k], data_y_pred_train):.2f}")

    # The mean squared error
    Ridge_rmse.append(sqrt(mse(data_y_test[k], data_y_pred_test)).real)
    print(f"RMSE Test : {sqrt(mse(data_y_test[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    Ridge_rsqr.append(r2_score(data_y_test[k], data_y_pred_test))
    print(f"R^2 Test: {r2_score(data_y_test[k], data_y_pred_test):.2f}")
    print()

    # LASSO reduction of model

    print(f"Running LASSO Reduction On The Model")
    # Make predictions using the testing set
    data_y_pred_test = l_regr.predict(data_X_test[k])
    data_y_pred_train = l_regr.predict(data_X_train[k])
    print("Coefficients:")
    print(l_regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y_train[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y_train[k], data_y_pred_train):.2f}")

    # The mean squared error
    Lasso_rmse.append(sqrt(mse(data_y_test[k], data_y_pred_test)).real)
    print(f"RMSE Test : {sqrt(mse(data_y_test[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    Lasso_rsqr.append(r2_score(data_y_test[k], data_y_pred_test))
    print(f"R^2 Test: {r2_score(data_y_test[k], data_y_pred_test):.2f}")
    print()

    # Elastic reduction of model

    print(f"Running Elastic Reduction On The Model")
    # Make predictions using the testing set
    data_y_pred_test = e_regr.predict(data_X_test[k])
    data_y_pred_train = e_regr.predict(data_X_train[k])
    print("Coefficients:")
    print(e_regr.coef_)

    # The mean squared error
    print(f"RMSE Train : {sqrt(mse(data_y_train[k], data_y_pred_train)).real:.2f}")
    # The mean squared error
    print(f"R^2 Train: {r2_score(data_y_train[k], data_y_pred_train):.2f}")

    # The mean squared error
    Elastic_rmse.append(sqrt(mse(data_y_test[k], data_y_pred_test)).real)
    print(f"RMSE Test : {sqrt(mse(data_y_test[k], data_y_pred_test)).real:.2f}")
    # The mean squared error
    Elastic_rsqr.append(r2_score(data_y_test[k], data_y_pred_test))
    print(f"R^2 Test: {r2_score(data_y_test[k], data_y_pred_test):.2f}")
    print()

# End of for loop

# Ridge scores
print(f"RMSE Mean of Ridge: {sum(Ridge_rmse)/len(Ridge_rmse):.2f}")
print(f"R^2 Mean of Ridge: {sum(Ridge_rsqr)/len(Ridge_rsqr):.2f}")
print()

# LASSO scores
print(f"RMSE Mean of LASSO: {sum(Lasso_rmse)/len(Lasso_rmse):.2f}")
print(f"R^2 Mean of LASSO: {sum(Lasso_rsqr)/len(Lasso_rsqr):.2f}")
print()

# Elastic scores
print(f"RMSE Mean of Elastic: {sum(Elastic_rmse)/len(Elastic_rmse):.2f}")
print(f"R^2 Mean of Elastic: {sum(Elastic_rsqr)/len(Elastic_rsqr):.2f}")
print()
