"""
Run comparision between this implementaion and scikit-learn
"""

# Author: Diana Chenyu Zhang <dczhang@uw.edu>

import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import linear_svm_squared_hinge_loss

def compare(x,y):
	"""Run this implementation and scikit-learn on the same dataset; print the results

    Parameters
    ----------
    x: features

    y: labels

    Returns
    -------
    error_train_lambda_opt: miscalssification error of this implementaion on training set
    error_test_lambda_opt: miscalssification error of this implementaion on testing set
    error_train_svm: miscalssification error of scikit-learn's function on training set
    error_test_svm: miscalssification error of scikit-learn's function on testing set
    """
	# Random train-test split and standardize the data
	x_train, x_test, y_train, y_test = train_test_split(
    	x, y, test_size=0.25, random_state=0)
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

    # Use this implementaion 
	lambduh = 1
	n = np.size(x_train, 1)
	eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(n-1, n-1), eigvals_only=True)[0]+lambduh)
	betas = linear_svm_squared_hinge_loss.mylinearsvm(lambduh, eta_init, 100, x_train, y_train)
	optimal_lambduh2 = linear_svm_squared_hinge_loss.find_optimal_lambduh(x_test,y_test,eta_init,-10,10,0.1)

	# Find optimal value of lambda through cross-validation
	betas_opt = linear_svm_squared_hinge_loss.mylinearsvm(optimal_lambduh2, eta_init, 100, x_train, y_train)
	error_train_lambda_opt = linear_svm_squared_hinge_loss.compute_misclassification_error(betas_opt[-1,:],x_train,y_train)
	error_test_lambda_opt = linear_svm_squared_hinge_loss.compute_misclassification_error(betas_opt[-1,:],x_test,y_test)
	# Print results using this implementation
	print('Using implementaion, misclassification error for the training set is: ', error_train_lambda_opt)
	print('Using implementaion, misclassification error for the testing set is: ', error_test_lambda_opt)


	# Use scikit-learn LinearSVC function with squared_hinge
	svm_l2 = svm.LinearSVC(penalty='l2', loss='squared_hinge')
	parameters = {'C':[10**i for i in range(-2, 2)]}
	clf_svm = GridSearchCV(svm_l2, parameters)
	clf_svm.fit(x_train, y_train)
	error_train_svm = 1 - clf_svm.score(x_train, y_train)
	error_test_svm = 1 - clf_svm.score(x_test, y_test)
	# Print results using scikit-learn
	print('Use sklearn, misclassification error for the training set is: ', error_train_svm)
	print('Use sklearn, misclassification error for the testing set is: ', error_test_svm)

	return error_train_lambda_opt, error_test_lambda_opt, error_train_svm, error_test_svm


