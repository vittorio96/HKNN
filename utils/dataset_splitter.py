import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, Y):
    
    x_train, x_val, y_train, y_val = train_test_split(X, Y, train_size=0.85, test_size=0.15, random_state=10)

    ## Make them as numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_train, x_val, y_train, y_val;
