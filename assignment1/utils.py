import numpy as np


def increase_order(mat: np.ndarray, order):
    print(f"Setting order to {order}")
    nr, nc = mat.shape
    new_cols = [[] for _ in range(nc * (order - 1))]

    for i in range(2, order + 1):
        for row in range(nr):
            for col in range(nc):
                val = mat[row][col] ** i
                new_cols[col + (nc * (i - 2))].append(val)

    if len(new_cols) != 0:
        new_cols = np.array(new_cols).T
        return np.hstack((mat, new_cols))

    return mat
