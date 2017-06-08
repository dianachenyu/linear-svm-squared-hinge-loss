# Linear Support Vector Machine with the Squared Hinge Loss

This code implement linear support vector machine with squared hinge loss. It uses fast-gradient descent algorithm. The step size is calculated using backtracking rule.

## Organization of the project
In the src folder, there are five .py files:

 - linear_svm_squared_hinge_loss.py implements the method, including training, visualizion, and printing out the results.
 - demo_simulated_data.py is a demo file, which launchs the method on a simple simulated dataset.
 - demo_real_world_data.py is a demo file, which launchs the method on a real-world dataset.
 - compare_to_sklearn.py compares the results from this implementation and scikit-learn.
 - test_compare.py is demo of the comparison.


## Installation 

### Data
For the real-world dataset, Spam dataset is used in the demo. It is from the book *The Elements of Statistical Learning*. You can access the dataset from [the website of the book](https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets).


### Required softwares and packages

The code is written Python 3. Beside python, you also need pip3 or conda to intall the following python packages:

- numpy
- pandas
- scipy
- matplotlib
- sklearn