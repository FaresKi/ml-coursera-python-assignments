# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np
def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    # =========================== YOUR CODE HERE =====================
    num_rows, num_cols = X_norm.shape
    for i in range(num_cols):
        mean =  X[:,i].mean()
        std =  X[:,i].std()
        mu[:,i] = mean
        sigma[:,i] = std
        X_norm[:,i] = (X[:,i] - mean)/std
    
    # ================================================================
    return X_norm, mu, sigma

data = np.loadtxt(os.path.join('/Users/fareskissoum/Documents/Personal-Projects.tmp/ml-coursera-python-assignments/Exercise1/Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)