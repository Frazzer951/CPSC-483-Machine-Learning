import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=2, linewidth=127)


# Load the dataset
df = pd.read_csv("Data1.csv")


data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()

# print(data_X)

centered_data_X = StandardScaler(with_std=False).fit_transform(data_X)
c_data_X_tran = centered_data_X.T

# print(c_data_X_tran)

# a)
cov_mat = np.cov(c_data_X_tran)
print("Covariance Matrix:")
print(cov_mat)

# b)
eig_vec, eig_val, _ = np.linalg.svd(cov_mat)
print("EigenVectors:")
print(eig_vec)
print("EigenValues:")
print(eig_val)

# c)
proj_mat = np.matmul(centered_data_X, eig_vec)
print("Projection Matrix:")
print(proj_mat)

# e)
total_sum = eig_val.sum()
exp_var = eig_val / total_sum
print("Explained Variance:")
print(list(exp_var))
