import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.array([5, 6, 7])
idxs = np.isin(a, b)
print(a[idxs])