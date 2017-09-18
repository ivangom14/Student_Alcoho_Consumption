import dateutil.parser
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA ,TruncatedSVD, NMF
from sklearn import decomposition

def normalize(dataset):
    norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return norm

def desnormalize(dataset):
    desnorm = dataset * (dataset.max() - dataset.min()) + dataset.min()
    return desnorm

def descomposicion(data_train, data_test, n_components=2 ,random_state=420):

    pca = decomposition.PCA(n_components=n_components, random_state= random_state)
    pca_train = pca.fit_transform(data_train)
    pca_test = pca.fit_transform(data_test)

    ica = decomposition.FastICA(n_components=n_components, random_state= random_state)
    ica_train = ica.fit_transform(data_train)
    ica_test = ica.fit_transform(data_test)

    tsvd = decomposition.TruncatedSVD(n_components=n_components, random_state= random_state)
    tsvd_train = tsvd.fit_transform(data_train)
    tsvd_test = tsvd.fit_transform(data_test)

    nmf = decomposition.NMF(n_components=n_components, random_state= random_state)
    nmf_train = nmf.fit_transform(data_train)
    nmf_test = nmf.fit_transform(data_test)

    for i in range(1, n_components + 1):
        data_train['pca_' + str(i)] = pca_train[:, i - 1]
        data_test['pca_' + str(i)] = pca_test[:, i - 1]

        data_train['ica_' + str(i)] = ica_train[:, i - 1]
        data_test['ica_' + str(i)] = ica_test[:, i - 1]

        data_train['tsvd_' + str(i)] = tsvd_train[:, i - 1]
        data_test['tsvd_' + str(i)] = tsvd_test[:, i - 1]

        data_train['nmf_' + str(i)] = nmf_train[:, i - 1]
        data_test['nmf_' + str(i)] = nmf_test[:, i - 1]

    return data_train, data_test

