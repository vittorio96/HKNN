import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, Y):
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.10, random_state=57)

    ## Make them as numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test;
