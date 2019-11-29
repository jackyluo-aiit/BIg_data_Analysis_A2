import numpy as np
import example
import matplotlib.pyplot as plt
from PIL import Image


def load_data(digit, num):
    data = example.load_data(digit, num)
    print("Load data:\n num_pic: ", num, "\n digit: ", digit)
    print('data size: ', np.shape(data))
    return data


def array_to_img(array):
    array = array * 255
    new_img = Image.fromarray(array.astype(np.uint8))
    return new_img


def show_image(data):
    # new_img = Image.new(new_type, (col * 28, row * 28))
    n = np.shape(data)[1]
    for i in range(n):
        #     each_img = array_to_img(np.array(data[:, i]).reshape(28, 28))
        #     each_img.show()
        #     # 第二个参数为每次粘贴起始点的横纵坐标。在本例中，分别为（0，0）（28，0）（28*2，0）依次类推，第二行是（0，28）（28，28），（28*2，28）类推
        #     new_img.paste(each_img, ((i * 28), 0))
        # return new_img
        img_data = np.reshape(data[:, i], (28, 28))
        plt.subplot(4, 5, i+1)
        plt.subplots_adjust(hspace= 0.5)
        plt.title('eigenvector %d'%(i+1))
        plt.imshow(np.real(np.reshape(img_data, (28, 28))))
        plt.gray()
    plt.show()


def pca(data, top_num_features=20):
    mean_vals = np.mean(data, axis=0)
    print(np.shape(mean_vals))
    mean_removed = data - mean_vals
    print(np.shape(mean_removed))
    cov_mat = np.cov(mean_removed, rowvar=False)
    print('cov_mat:', np.shape(cov_mat))
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    print('eigenvalues:', np.shape(eigenvalues), 'eigenvectors:', np.shape(eigenvectors))
    # eigen_sorted_index = np.argsort(-eigenvalues)
    # eig_val_index = eigen_sorted_index[:-(top_num_features + 1): -1]
    # print(eigen_sorted_index)
    # eigen_sorted_index = eigen_sorted_index[:top_num_features]
    eig_val_index = np.argsort(eigenvalues)
    eigen_sorted_index= eig_val_index[:-(top_num_features + 1): -1]
    print(eigen_sorted_index)
    # print(eig_val_index)
    selected_eig_vects = eigenvectors[:, eigen_sorted_index]
    print("selected eigenvectors:", np.shape(selected_eig_vects))
    return eigenvalues, selected_eig_vects, mean_vals


data = load_data(0, 5000)
eig_values, top_num_eig_vects, mean_image = pca(data)
example.display(mean_image)
show_image(top_num_eig_vects)
# eig_val_index = np.argsort(eig_values)
# top_100_eig_index = eig_val_index[:-(100 + 1): -1]
# plt.plot(np.arange(100), eig_values[top_100_eig_index])
# plt.show()
