"""
Demo of the method on a real-world dataset
"""

# Author: Diana Chenyu Zhang <dczhang@uw.edu>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import linear_svm_squared_hinge_loss

# Use the real-world datset, Spam dataset from book The Elements of Statistical Learning
spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',
                               header=None)
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1 

# Use the train-test split inidcator provided along with the dataset
test_indicator = np.array(test_indicator).T[0]
x_train = x[test_indicator == 0, :]
x_test = x[test_indicator == 1, :]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Use lambda = 1 first
lambduh = 1
n = np.size(x, 1)
eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(n-1, n-1), eigvals_only=True)[0]+lambduh)
betas = linear_svm_squared_hinge_loss.mylinearsvm(lambduh, eta_init, 100, x_train, y_train)

# Calculate misclassifcation error on training set and testing set
error_train_lambda1 = linear_svm_squared_hinge_loss.compute_misclassification_error(betas[-1,:],x_train,y_train)
error_test_lambda1 = linear_svm_squared_hinge_loss.compute_misclassification_error(betas[-1,:],x_test,y_test)
print('Misclassification error for the lambda value 1 on the training set is: ', error_train_lambda1)
print('Misclassification error for the lambda value 1 on the testing set is: ', error_test_lambda1)

# Plot miscalssification error and objective value
linear_svm_squared_hinge_loss.plot_misclassification_error(betas, x_train, y_train,
                             title='Training set misclassification error when lambda = 1',
                             file_name = 'misclass_plot_train_lambda1.png')
linear_svm_squared_hinge_loss.plot_misclassification_error(betas, x_test, y_test,
                             title='Test set misclassification error when lambda = 1',
                             file_name = 'misclass_plot_test_lambda1.png')
linear_svm_squared_hinge_loss.plot_objective(betas, lambduh, x_train, y_train, file_name = 'objective_plot_train_lambda1.png' )
 

# Find optimal value of lambda through cross-validation
optimal_lambduh1 = linear_svm_squared_hinge_loss.find_optimal_lambduh(x_test,y_test,eta_init,-1000,1000,10)
print('Optimal value of lambda is: ', optimal_lambduh1)
optimal_lambduh2 = linear_svm_squared_hinge_loss.find_optimal_lambduh(x_test,y_test,eta_init,-10,10,0.01)
print('Optimal value of lambda is: ', optimal_lambduh2)

# Calculate misclassifcation error on training set and testing set
betas_opt = linear_svm_squared_hinge_loss.mylinearsvm(optimal_lambduh2, eta_init, 100, x_train, y_train)
error_train_lambda_opt = linear_svm_squared_hinge_loss.compute_misclassification_error(betas_opt[-1,:],x_train,y_train)
error_test_lambda_opt = linear_svm_squared_hinge_loss.compute_misclassification_error(betas_opt[-1,:],x_test,y_test)
print('Misclassification error for the optimal lambda value on the training set is: ', error_train_lambda_opt)
print('Misclassification error for the optimal lambda value on the testing set is: ', error_test_lambda_opt)

# Plot miscalssification error and objective value
linear_svm_squared_hinge_loss.plot_misclassification_error(betas_opt, x_train, y_train,
                             title='Training set misclassification error for the optimal lambda value.',
                             file_name = 'misclass_plot_train_lambda_opt.png')
linear_svm_squared_hinge_loss.plot_misclassification_error(betas_opt, x_test, y_test,
                             title='Test set misclassification error for the optimal lambda value.',
                             file_name = 'misclass_plot_test_lambda_opt.png')
linear_svm_squared_hinge_loss.plot_objective(betas_opt, optimal_lambduh2, x_train, y_train, file_name = 'objective_plot_train_lambda_opt.png')
