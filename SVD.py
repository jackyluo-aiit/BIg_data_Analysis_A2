import numpy as np
np.set_printoptions(suppress=True)


M = np.mat(np.array([[1,2,3],
              [3,4,5],
              [5,4,3],
              [0,2,4],
              [1,3,5]]))
print(M)
M_mean = M.mean(axis=0)
print(np.shape(M_mean))
M_mean_removed = M - M_mean
U, s, V = np.linalg.svd(M_mean_removed)
print('U: ',U)
print('s: ',s)
print('V: ',V)
