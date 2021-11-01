import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# All functions used inside preprocess_dataset

def na_per_obs(tx, nobs, name="Tx"):
    """ Investigates how many missing (n/a) data exists in the dataset for each row, i.e. observation
        and prints out the statistics """
    print("{nm}: # of Unavailable Data for each Observation:".format(nm=name))
    mask = (tx == -999)
    temp = np.sum(mask, axis=1)
    res = temp[temp > 0]
    print(
        "Number of Observations with at least one unavailable data: {l}, which is {rat}% of all data".format(l=len(res),
                                                                                                             rat=100 * len(
                                                                                                                 res) / nobs))

def na_per_feature(tx, nobs, name="Tx"):
    """ Investigates how many missing (n/a) data exists in the dataset for each column, i.e. feature
        and prints out the statistics """
    # print("\n{nm}: # of Unavailable Data for each Feature:".format(nm=name))
    mask = (tx == -999)
    temp = np.sum(mask, axis=0)
    nafeats = temp[temp > 0]
    ratios = 100 * nafeats / nobs
    # print(ratios)
    print(name + "# of features with unavailable data: {l}".format(l=len(nafeats)))
    print(name + "# of features with unavailable data more than 70%: {l}".format(l=sum(ratios > 70)))
    # print("*" * 100)

def remove_na_feature(tx, features, nobs):
    # Removes features with n/a's more than 70% of the size of dataset
    mask = (tx == -999)
    temp = np.sum(mask, axis=0)
    cols_removed = np.argwhere((temp * 100 / nobs) > 70)
    return np.delete(tx, cols_removed, 1), cols_removed

def imputation(tx, name="median"):
    """ Replacing all n/a data by the chosen statistics of each column, i.e. feature
        and returns the up-to-date input matrix. Statistics can be either median, mod or mean """
    mask = (tx == -999)
    temp = np.sum(mask, axis=0)
    inds = np.argwhere(temp)
    for i, ind in enumerate(inds):
        feat = tx[:,ind]
        feat_na = feat[feat != -999]
        if name == "median":
            feat[feat == -999] = np.median(feat_na)
        elif name == "mod":
            vals, counts = np.unique(feat_na, return_counts=True)
            j = np.argmax(counts)
            feat[feat == -999] = vals[j]
        else:
            feat[feat == -999] = np.mean(feat_na)
        tx[:,ind] = feat
    return tx

def find_categorical_data(tx, max_unique):
    """ Finds the number of unique values for each column less than max_unique value
        to find out categorical data and returns the indices """
    inds = []
    nouns = []
    for i in range(tx.shape[1]):
        val = len(np.unique(tx[:, i]))
        if (val < max_unique):
            inds.append(i)
            nouns.append(val)
    return inds, nouns

def handle_categorical_feature(tx, cat_inds, feats=[]):
    """ Applies one-hot encoding to input matrix and returns the up-to-date
        input matrix and modifies features names vector"""
    for ind in cat_inds:
        feat_vec = tx[:, ind]
        f = feats[ind]
        a = np.unique(feat_vec)
        for j in a:
            y = np.expand_dims((feat_vec == j).astype(int), axis=0)
            tx = np.concatenate((tx, y.T), axis=1)
            feats = np.append(feats, f + "_" + str(int(j)))
    return np.delete(tx, cat_inds, 1), np.delete(feats, cat_inds)

def handle_categorical_feature_testdataset(tx, cat_inds):
    """ Applies one-hot encoding to input matrix and returns the up-to-date
        input matrix without modifying features names vector"""
    for ind in cat_inds:
        feat_vec = tx[:, ind]
        a = np.unique(feat_vec)
        for j in a:
            y = np.expand_dims((feat_vec == j).astype(int), axis=0)
            tx = np.concatenate((tx, y.T), axis=1)
    return np.delete(tx, cat_inds, 1)


def handle_outliers(x, ncats):
    """ Applies handling outliers technique for each feature and returns
        the up-to-date input matrix """
    x_1 = x.copy()
    x_2 = x.copy()
    for i in range(x.shape[1]-ncats):
        q1, q3 = np.percentile(x_1[:, i][x_1[:, i].argsort()], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        x_2[:, i][x_2[:, i] < lower_bound] = lower_bound
        x_2[:, i][x_2[:, i] > upper_bound] = upper_bound
    return x_2

def build_poly(tx, degree, n_feat):
    """ Applies feature augmentation by polynomial extension with respect 
        to the given degree and returns the up-to-date input matrix """
    deg = np.arange(degree)+1
    old_ncol = tx.shape[1] - n_feat
    tx_part = tx[:,0:old_ncol]
    res = np.zeros((tx.shape[0], old_ncol*degree+n_feat))
    new_cat_inds = np.arange(n_feat) + old_ncol
    for i in deg:
        if (i==1):
            res[:,0:old_ncol+n_feat] = tx
        else:
            res[:,(i-1)*old_ncol+n_feat:old_ncol*i+n_feat] = np.power(tx_part,i)
    return res, new_cat_inds

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def standardize_dataset(tx, cat_inds):
    ms = []
    stds = []
    for ind in range(tx.shape[1]):
        if ind not in cat_inds:
            tx[:,ind], mu, std = standardize(tx[:,ind])
            ms.append(mu)
            stds.append(std)
        else:
            ms.append(0)
            stds.append(1)
    return tx, ms, stds

"""def standardize_dataset(tx, nouns):"""
""" Finds means and standard deviation of each feature except for categorical
        data, applies standardization and returns the up-to-date input vector with 
        means and standard_deviations """
"""
    ms = []
    stds = []
    for ind in range(tx.shape[1]):
        if ind <tx.shape[1]-nouns:
            tx[:,ind], mu, std = standardize(tx[:,ind])
            ms.append(mu)
            stds.append(std)
        else:
            ms.append(0)
            stds.append(1)
    return tx, ms, stds
"""
def add_feat_offset(tx):
    """ Adds constant feature to input matrix and returns it """
    nobs = tx.shape[0]
    return np.concatenate((tx, np.ones((nobs, 1))), axis=1)

def statistic_feats(tx):
    """ Checks whether the columns of input matrix is standardized or not,
        prints out the means and standard deviations """
    means = np.mean(tx, axis=0)
    stds = np.std(tx, axis =0)
    print(means)
    print(stds)
    

def standardize_test_dataset(tx, ms, stds, cat_inds):
    """ Applies standardization according to the means and standard deviations obtained 
        from training dataset and returns the up-to-date input vector """
    for ind in range(tx.shape[1]):
        if ind not in cat_inds:
            temp = tx[:,ind]
            tx[:,ind] = (temp - ms[ind])/stds[ind]
    return tx


# Complete Functions to Implement Data Preprocessing, Feature Engineering
"""
def preprocess_dataset(y, tX, features, degree, rem_inds):

    nobs = len(y)

    # Handling Missing Values
    tX_r, cols_removed = remove_na_feature(tX, features, nobs)  
    features_r = np.delete(features, cols_removed)
    tX_rm = imputation(tX_r, "median")

    # Removing Dependent Features
    tX_rm = np.delete(tX_rm, rem_inds, 1)
    features_r = np.delete(features_r, rem_inds)

    # Handling Categorical Data
    cat_inds, nouns = find_categorical_data(tX_rm, 100)
    tX_rmc, feats_rmcat = handle_categorical_feature(tX_rm, cat_inds, features_r)

    # Handling Outliers --- It makes worse!!
    tX_rmch = handle_outliers(tX_rmc, np.sum(nouns))

    # Standardization of Data
    #tX_rmc, ms, stds = standardize_dataset(tX_rmch, sum(nouns))
    tX_rmc = tX_rmch.copy()
    tX_rmc[:,0:tX_rmc.shape[1]-sum(nouns)], ms, stds = standardize(tX_rmc[:,0:tX_rmc.shape[1]-sum(nouns)])


    # Feature Augmentation by Polynomial Degree
    tX_rmcp, new_cat_inds = build_poly(tX_rmc, degree, sum(nouns))

    # Adding Feature for Offset Term
    tX_rmcpo = add_feat_offset(tX_rmcp)

    return y, tX_rmcpo, ms, stds
"""


def preprocess_dataset(y, tX, features, degree, rem_inds):
    nobs = len(y)

    # Handling Missing Values
    features_r = features
    tX_rm = imputation(tX, "median")

    # Removing Dependent Features
    tX_rm = np.delete(tX_rm, rem_inds, 1)
    features_r = np.delete(features_r, rem_inds)

    # Handling Categorical Data
    cat_inds, nouns = find_categorical_data(tX_rm, 100)
    tX_rmc, feats_rmcat = handle_categorical_feature(tX_rm, cat_inds, features_r)

    # Feature Augmentation by Polynomial Degree
    tX_rmcp, new_cat_inds = build_poly(tX_rmc, degree, sum(nouns))

    # Standardization of Data
    tX_rmcps, ms, stds = standardize_dataset(tX_rmcp, new_cat_inds)

    # Adding Feature for Offset Term
    tX_rmcpso = add_feat_offset(tX_rmcps)

    return y, tX_rmcpso, ms, stds



def preprocess_test_dataset(tX, degree, ms, stds, rem_inds, rem_cols = []):
    """ Preprocess the raw input and output test data according to some data 
        obtained from preprocessing of train dataset, returns the preprocessed ones """

    tX_r = np.delete(tX, rem_cols, 1)
    tX_rm = imputation(tX_r, 'median')
    tX_rm = np.delete(tX_rm, rem_inds, 1)
    cat_inds, nouns = find_categorical_data(tX_rm, 100)
    tX_rmc = handle_categorical_feature_testdataset(tX_rm, cat_inds)
    tX_rmcp, new_cat_inds = build_poly(tX_rmc, degree, sum(nouns))
    tX_rmcps = standardize_test_dataset(tX_rmcp, ms, stds, new_cat_inds)
    tX_rmcpso = add_feat_offset(tX_rmcps)

    return tX_rmcpso
