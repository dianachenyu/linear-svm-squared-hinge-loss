# Linear Support Vector Machine with the Squared Hinge Loss

This code implement linear support vector machine with squared hinge loss. It uses fast-gradient descent algorithm. The initial step size is set using the Lippshitz function and then optimizing the step size using the backtracking rule.

## Organization of the project
In the src folder, there are five .py files:

 - linear_svm_squared_hinge_loss.py implements the method.
 - demo_simulated_data.py is a demo file, which launchs the method on a simple simulated dataset.
 - demo_real_world_data.py is a demo file, which launchs the method on a real-world dataset.
 - compare_to_sklearn.py runs comparison between my own implementation and scikit-learn 
 - test_compare.py is demo of the comparison


## Installtion 

### Data
For the real-world dataset, Spam dataset, used in the exmaple, it is from the book The Elements of Statistical Learning. You can access the dataset here:
```
https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data

```
### Required software and packages

The code is written Python 3. Beside python, you also need pip3 or conda to intall the following python packages:

- numpy
- pandas
- scipy
- matplotlib
- sklearn