import os.path
import numpy as np
import matplotlib.pyplot as plt

def display(Xrow):
    ''' Display a digit by first reshaping it from the row-vector into the image.  '''
    plt.imshow(np.reshape(Xrow,(28,28)))
    plt.gray()
    plt.show()


def load_data(digit=0, num=5000):
    ''' 
    Loads all of the images into a data-array (for digits 0 through 5). 

    The training data has 5000 images per digit and the testing data has 200, 
    but loading that many images from the disk may take a while.  So, you can 
    just use a subset of them, say 200 for training (otherwise it will take a 
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and 
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.
    
    '''
    X = np.zeros((num, 784),dtype=np.uint8)   #784=28*28
    print('\nReading digit %d' % digit)
    for i in range(num):
        if not i%100: print(i,'.')
        pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
        with open(pth, 'rb') as infile:
            header = infile.readline()
            header2 = infile.readline()
            header3 = infile.readline()
            image = np.fromfile(infile, dtype=np.uint8).reshape(1, 784)
        X[i,:] = image
    print('\n')
    return X

if __name__ == '__main__':
    X = load_data(1)
    display(X[4999, :])