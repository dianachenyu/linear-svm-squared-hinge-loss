"""
Demo of the method on a simulated dataset
"""

# Author: Diana Chenyu Zhang <dczhang@uw.edu>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import linear_svm_squared_hinge_loss

# Simulate a dataset with  60 obervations and 50 features
features = np.zeros((60, 50))
features[0:30, :] = np.random.normal(scale=1, size=(30, 50))
features[30:60, :] = np.random.normal(loc=1, scale=5, size=(30, 50))
labels = np.asarray([1]*30 + [-1]*30)

# Random train-test split. Test set contain 75% of the data.
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=0)

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# Use lambda = 1 first
lambduh = 1
n = np.size(x_train, 1)
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
optimal_lambduh2 = linear_svm_squared_hinge_loss.find_optimal_lambduh(x_test,y_test,eta_init,-100,100,0.1)
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
