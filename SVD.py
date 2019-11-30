import numpy as np
np.set_printoptions(suppress=True)


M = np.mat(np.array([[1,2,3],
              [3,4,5],
              [5,4,3],
              [0,2,4],
              [1,3,5]]))
print('M:\n',M)
M_t_M = M.T.dot(M)
M_M_t = M.dot(M.T)
print('M_t_M:\n', M_t_M)
print('M_M_t:\n', M_M_t)
M_mean = M.mean(axis=0)
M_mean_removed = M - M_mean
eig_vals, eig_vects = np.linalg.eig(M_t_M)
print('eig_vals:\n', eig_vals, '\neig_vects:\n', eig_vects)
eig_vals, eig_vects = np.linalg.eig(M_M_t)
print('eig_vals:\n', eig_vals, '\neig_vects:\n', eig_vects)
U, s, V = np.linalg.svd(M_mean_removed)
print('U:\n',U)
print('s:\n',s, 'shape: ', np.shape(s))
print('V:\n',V)
sigma_mat = np.zeros([3,3])
for i in range(3):
    sigma_mat[i][i] = s[i]
print('retained energy:', np.square(s[0])/(np.square(s[0])+np.square(s[1])))
M_recoved = U[:,:1].dot(sigma_mat[:1,:1]).dot(V[:1,:]) + M_mean
print('recoved M:\n', M_recoved)


