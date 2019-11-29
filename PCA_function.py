import numpy as np
import example


def load_data(digit, num):
    data = example.load_data(digit,num)
    print("Load data:\n num_pic: ", num,"\n digit: ", digit)
    print('data size: ', np.shape(data))
    return data


def pca(data, top_num_features=20):
    mean_vals = np.mean(data, axis=0)
    print(np.shape(mean_vals))
    mean_removed = data - mean_vals
    print(np.shape(mean_removed))
    cov_mat = np.cov(mean_removed, rowvar=False)
    print('cov_mat:', np.shape(cov_mat))
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    print('eigenvalues:', np.shape(eigenvalues), 'eigenvectors:',np.shape(eigenvectors))
    eigen_sorted_index = np.argsort(-eigenvalues)
    # eig_val_index = eigen_sorted_index[:-(top_num_features + 1): -1]
    # print(eigen_sorted_index)
    eigen_sorted_index = eigen_sorted_index[:top_num_features]
    print(eigen_sorted_index)
    # eig_val_index = np.argsort(eigenvalues)
    # eig_val_index = eig_val_index[:-(top_num_features + 1): -1]
    # print(eig_val_index)
    selected_eig_vects = eigenvectors[:,]


data = load_data(0, 5000)
pca(data)
