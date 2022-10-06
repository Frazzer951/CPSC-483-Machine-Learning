import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=2, linewidth=127)
pd.options.display.float_format = "{:,.2f}".format

# Load the dataset
df = pd.read_csv("Data1.csv")


data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()

centered_data_X = StandardScaler().fit_transform(data_X)
c_data_X_tran = centered_data_X.T


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


# d)
r, c = eig_vec.shape
coef_mat = pd.DataFrame(eig_vec, columns=[f"PC{n+1}" for n in range(c)], index=["T", "P", "TC", "SV"])
coef_mat = coef_mat.sort_values(by=["PC1"], ascending=False)
print("Principal Components Matrix:")
print(coef_mat)


# e)
total_sum = eig_val.sum()
exp_var = eig_val / total_sum
print("Explained Variance:")
print(list(exp_var))


# f)
cor_mat = np.corrcoef(c_data_X_tran, proj_mat.T)
print("Correlation Matrix:")
print(cor_mat)
