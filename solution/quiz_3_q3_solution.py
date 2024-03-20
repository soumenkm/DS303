# Question 3
# Part 3
from sklearn.model_selection import KFold
from sklearn import ensemble
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def cross_validate(X_train, Y_train, cands1, cands2):
    """Hint 1: Use KFold class 
    Hint 2: Use ensemble.RandomForestClassifier model 
    Hint 3: Calculate accuracy and 
    return the list of accuracies
    """
    # Instantiate the KFold class
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Initialize the array to store the accuracy values
    cv_scores = np.zeros((kf.get_n_splits(), len(cands1), len(cands2))) # (5,3,3)
    # Iterate over the splits
    for i, (train_index, test_index) in enumerate(kf.split(Y_train)):
        # Get the train and test splits of the data
        X, X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y, y_test = Y_train[train_index], Y_train[test_index]
        # Iterate over the possible values of cands1 which is the max_depth
        for j, max_depth in enumerate(cands1):
            # Iterate over the possible values of cands1 which is the n_estimators
            for k, n_estimators in enumerate(cands2):
                # Train a random forest with every possible combination of the hyperparameters
                clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                clf.fit(X, y)
                # Store the accuracy obtained on the test set
                cv_scores[i,j,k] = clf.score(X_test, y_test)
    return cv_scores

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.Series(iris.target)

# Example usage:
cands1 = [5, 10, 15]  # max_depth candidates
cands2 = [50, 100, 150]  # n_estimators candidates

# Run the function
output = cross_validate(X, Y, cands1, cands2)

# Sample output
print("Cross-validation scores:", output)
