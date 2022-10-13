import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2, linewidth=127)
pd.options.display.float_format = "{:,.2f}".format

# Load the dataset
df = pd.read_csv("Data1.csv")


data_y = df.drop(columns=["T", "P", "TC", "SV"]).to_numpy().T[0]
data_X = df.drop(columns="Idx").to_numpy()


centered_data_X = StandardScaler().fit_transform(data_X)
c_data_X_tran = centered_data_X.T

# Part 1 Starts here
print("Part 1")

# pt1 a)
cov_mat = np.cov(c_data_X_tran)
print("\nCovariance Matrix:")
print(cov_mat)


# pt1 b)
eig_vec, eig_val, _ = np.linalg.svd(cov_mat)
print("\nEigenVectors:")
print(eig_vec)
print("\nEigenValues:")
print(eig_val)


# pt1 c)
proj_mat = np.matmul(centered_data_X, eig_vec)
print("\nProjection Matrix:")
print(proj_mat)


# pt1 d)
r, c = eig_vec.shape
coef_mat = pd.DataFrame(eig_vec, columns=[f"PC{n+1}" for n in range(c)], index=["T", "P", "TC", "SV"])
coef_mat = coef_mat.sort_values(by=["PC1"], ascending=False)
print("\nPrincipal Components Matrix:")
print(coef_mat)


# pt1 e)
total_sum = eig_val.sum()
exp_var = eig_val / total_sum
print("\nExplained Variance:")
print(list(exp_var))


# pt1 f)
combined_data = np.hstack((centered_data_X, proj_mat))
cor_mat = np.corrcoef(combined_data.T)
print("\nCorrelation Matrix:")
print(cor_mat)


# Part 2 Starts here
print("\n\nPart 2")

# pt2 a)
num_to_keep = 0
threshhold = 0.8
sum_sofar = 0
while sum_sofar < threshhold:
    sum_sofar += exp_var[num_to_keep]
    num_to_keep += 1
pcs_to_keep = [f"PC{n+1}" for n in range(num_to_keep)]
ev80 = coef_mat[pcs_to_keep]
print("\nExplained Variance > 80%:")
print(f"Keep first {num_to_keep} components")
print(ev80)

# pt2 b)
eig_avg = eig_val.sum() / len(eig_val)
kaiser_keep = [val >= eig_avg for val in eig_val]
kaiser = coef_mat
for i, bool in enumerate(kaiser_keep):
    if not bool:
        kaiser.drop(columns=[f"PC{i+1}"], inplace=True)
print("\nKaiser Criteria")
print(kaiser_keep)
print(kaiser)
