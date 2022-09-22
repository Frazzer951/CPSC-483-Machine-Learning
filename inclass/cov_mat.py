import numpy as np

np.set_printoptions(precision=2, linewidth=127)

data = np.array(
    [
        [-3, -4],
        [-1, -2],
        [-1, 0],
        [1, 0],
        [1, 2],
        [3, 4],
    ]
).T

cov_mat = np.cov(data)

print(cov_mat)

u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)

print(u)
print(s)
# print(vh)
