# Question 1:
# Part 1a: entropy
def entropy(y: np.ndarray) -> np.float64:
    """
    Calculate the entropy of a given set of labels.

    Parameters:
    y (array-like): Labels array.

    Returns:
    float: Entropy value.
    """
    # Begin your code here
    hist = np.bincount(y)
    ps = hist/len(y)
    e = - np.sum([p * np.log2(p) for p in ps if p > 0])
    # End your code here
    return e

# Part 1b: randomforestclassifier.fit
def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Fit the random forest classifier to the training data.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Target values.

    Returns:
    None
    """
    # Begin your code here
    dummy_data = X_train.copy()
    dummy_data['target'] = y_train
    
    self.tree_list = []
    
    for i in range(self.n_estimators):
        
        if self.bootstrap == True:
            df = self.row_sampling(dummy_data, self.max_samples)
        else:
            df = dummy_data.copy()
        
        tree = DecissionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
        
        tree.fit(df.drop('target', axis=1), df.target)

        self.tree_list.append(tree)
    # End your code here
    return None

# Part 1c: randomforestclassifier.predict
def predict(self, X_test: pd.DataFrame) -> list:
    """
    Make predictions using the fitted random forest classifier.

    Parameters:
    X_test (array-like): Test data features.

    Returns:
    array-like: Predicted target values.
    """
    # Begin your code here
    y_preds = np.empty((X_test.shape[0], len(self.tree_list)))
    # Let each tree make a prediction on the data
    for i, tree in enumerate(self.tree_list):
        # Indices of the features that the tree has trained on
        # idx = tree.feature_indices
        # Make a prediction based on those features
        prediction = tree.predict(X_test)
        y_preds[:, i] = prediction
    
    y_pred = []
    # For each sample
    for sample_predictions in y_preds:
        # Select the most common class prediction
        y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
    
    print(type(X_test))
    print(type(y_pred))
    # End your code here
    return y_pred

# Part 2:
# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Creating a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Making predictions on the train and test sets
y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

# Calculating accuracy for train and test sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Part 3:
"""
The issue lies in lines 15 and 16, where the arrays X and y were shuffled separately. 
Consequently, the relationship between the features and their corresponding labels was disrupted. 
In other words, the y values associated with specific features were shuffled independently, leading to mismatched pairs.

To address this problem, we can resolve it by shuffling the data jointly using shuffle=True (in line 13) when splitting the dataset, 
ensuring that the correspondence between the features and labels is maintained.
"""
# np.random.shuffle(X) # Bugs
# np.random.shuffle(y) # Bugs
X, y,coef = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                       random_state=None, shuffle=True,noise=20,coef=True) # Solution
