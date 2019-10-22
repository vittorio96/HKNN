import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduce_dimensionality(original_feature_set):
    """
        :type original_feature_set: numpy.ndarray
        :rtype: x: numpy.ndarray
    """

    original_feature_set = StandardScaler(copy=True, with_mean=True, with_std=False).fit_transform(original_feature_set)
    pca = PCA(n_components=0.9, svd_solver='full')
    principal_components = pca.fit_transform(original_feature_set)
    x = pd.DataFrame(data=principal_components)
    return x.values


def prepare_dataset(datasets, csv_name):

    ## Load Iris dataset
    df = pd.read_csv('/Users/vittoriodenti/Dev/Software/HKNN/datasets/' + datasets[csv_name]['name'])

    ## Separating out the target
    target = df[datasets[csv_name]['target']].values

    ## Separating out the features
    to_drop_list = [datasets[csv_name]['target']]+ datasets[csv_name]['to_drop']
    features_set = df.drop(to_drop_list, axis = 1).values

    ## Apply PCA
    x = reduce_dimensionality(features_set)

    ## Encode the name of the classes into numbers
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(target)

    ## Transform X and Y into numpy arrays
    X = np.array(x.tolist())
    Y = np.array(Y)

    return X, Y