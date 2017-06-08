import numpy as np
import pandas as pd
from compare_to_sklearn import compare

features = np.zeros((60, 50))
features[0:30, :] = np.random.normal(scale=1, size=(30, 50))
features[30:60, :] = np.random.normal(loc=1, scale=5, size=(30, 50))
labels = np.asarray([1]*30 + [-1]*30)

print('Comparison on a simulated dataset: ')
error_train_ipl, error_test_ipl, error_train_svm, error_test_svm = compare(features,labels)



spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1 

print('Comparison on a real-world dataset: ')
error_train_ipl2, error_test_ipl2, error_train_svm2, error_test_svm2 = compare(x,y)
