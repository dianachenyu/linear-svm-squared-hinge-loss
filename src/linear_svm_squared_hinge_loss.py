"""
Linear Support Vector Machine with the Squared Hinge Loss
"""

# Author: Diana Chenyu Zhang <dczhang@uw.edu>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing


def compute_grad(beta, lambduh, x, y):
    """Computes gradient of ojbective funtion.

    Parameters
    ----------
    beta: beta parameter

    lambduh: lambda parameter

    x: features

    y: labels

    Returns
    -------
    gradient of ojbective funtion.
    """
    yx = y[:, np.newaxis]*x
    yxbeta =yx.dot(beta)
    compare_result= np.maximum(0, 1-yxbeta)
    part1 = -2/len(y)*compare_result.dot(yx)
    part2 = 2*lambduh*beta      
    return (part1+part2)


def compute_objective(beta, lambduh, x, y):
    """Computes ojbective value.

    Parameters
    ----------
    beta: beta parameter

    lambduh: lambda parameter

    x: features

    y: labels

    Returns
    -------
    ojbective value
    """
    yx = y[:, np.newaxis]*x
    yxbeta =yx.dot(beta)
    compare_result= np.maximum(0, 1-yxbeta)
    part1 = 1/len(y) *(np.sum(compare_result**2))
    part2 = lambduh * np.linalg.norm(beta)**2
    return (part1+part2)


def backtracking(beta, lambduh, eta, x, y, alpha=0.5, betaparam=0.8,
                   maxiter=100):
    """ Use backtracking rule to update step size.

    Parameters
    ----------
    beta: beta parameter

    lambduh: lambda parameter
    
    eta: prior step size

    x: features

    y: labels

    Returns
    -------
    eta: updated step size
    """
    grad_beta = compute_grad(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = False
    iter = 0
    while (not found_eta) and iter < maxiter:
        if compute_objective(beta - eta * grad_beta, lambduh, x=x, y=y) < compute_objective(beta, lambduh, x=x, y=y) - alpha * eta * norm_grad_beta ** 2:
            found_eta = True
        else:
            eta *= betaparam
            iter += 1
    return eta


def fast_grad(beta_init, theta_init, lambduh, eta_init, maxiter, x, y):
    """ Fast gradient algorithm.

    Parameters
    ----------
    beta_init: initial beta values

    theta_init: initial theta values

    lambduh: lambda parameter
    
    eta_init: init step size

    maxiter: maximum number of iterations

    x: features

    y: labels

    Returns
    -------
    beta_vals: beta values of all iterations

    theta_vals: theta values of all iterations
    """
    beta = beta_init
    theta = theta_init
    grad = compute_grad(theta, lambduh, x=x, y=y)
    beta_vals = beta
    theta_vals = theta
    iter = 0
    while iter < maxiter:
        eta = backtracking(theta, lambduh, eta=eta_init, x=x, y=y)
        beta_new = theta - eta*grad
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        grad = compute_grad(theta, lambduh, x=x, y=y)
        beta = beta_new
        iter += 1
    return beta_vals, theta_vals


def mylinearsvm(lambduh, eta_init, maxiter, x, y):
    """ Linear Support Vector Machine

    Parameters
    ----------
    lambduh: lambda parameter
    
    eta_init: init step size

    maxiter: maximum number of iterations

    x: features

    y: labels

    Returns
    -------
    beta: beta values of all iterations
    """    
    d = np.size(x, 1)
    beta_init = np.zeros(d)
    theta_init = np.zeros(d)
    betas, thetas = fast_grad(beta_init, theta_init, lambduh, eta_init, maxiter,x,y)
    return betas 


def compute_misclassification_error(beta_opt, x, y):
    """ Compute misclassication error
    Parameters
    ----------
    beta_opt: beta values

    x: features

    y: labels

    Returns
    -------
    Fraction of misclassication
    """
    y_pred = x.dot(beta_opt) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1, same as logistic regression
    return np.mean(y_pred != y)


def plot_misclassification_error(betas, x, y, title='', file_name =''):
    """ Plot misclassication error
    
    Parameters
    ----------
    beta_opt: beta values

    x: features

    y: labels

    title: title of the plot

    file_name: save the plot with the name

    Returns
    -------
    Plot. x-axis: interation
          y-axis: misclassication error
    """
    iter = np.size(betas, 0)
    errors = np.zeros(iter)
    for i in range(iter):
        errors[i] = compute_misclassification_error(betas[i, :], x, y)
    fig, ax = plt.subplots()
    ax.plot(range(1, iter + 1), errors, c='red')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    if title:
        plt.title(title)
    if file_name:
        plt.savefig(file_name)
    plt.show()


def plot_objective(betas, lambduh, x, y, file_name =''):
    """ Plot objective values

    Parameters
    ----------
    betas: beta values of all iterations

    x: features

    y: labels

    file_name: save the plot with the name

    Returns
    -------
    Plot. x-axis: interation
          y-axis: objectve values
    """
    iter = np.size(betas, 0)
    objs = np.zeros(iter)
    for i in range(0, iter):
        objs[i] = compute_objective(betas[i, :], lambduh, x=x, y=y)
    fig, ax = plt.subplots()
    ax.plot(objs)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value for linear support vector machine when lambda='+str(lambduh))
    if file_name:
        plt.savefig(file_name)
    plt.show()


def find_optimal_lambduh(x,y,eta_init,min,max,step):
    """ Find lambda value through cross-validation

    Parameters
    ----------
    x: features

    y: labels

    eta_init: initial step size

    min: minimum lambda value to be searched 

    max: maximum lambda value to be searched 

    step: step size for the search

    Returns
    -------
    opt_lambduh: lambda gives a smallest misclassification error
    """
    lambduh_list = np.arange(min,max,step)
    mis_error = np.zeros(len(lambduh_list))
    i = 0
    for lambduh in lambduh_list:
        betas = mylinearsvm(lambduh, eta_init,100,x,y)
        mis_error[i] = compute_misclassification_error(betas[-1,:],x,y) 
        i= i+1
    opt_lambduh = lambduh_list[np.argmin(mis_error)]
    return opt_lambduh