# Question 2
# Part 1a: Adaboost.fit
def fit(self, X, y):
    """Fit the Adaboost model to the training data.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns:
    --------
    None
    """
    # Write your code here
    n_samples, n_features = np.shape(X)

    # Initialize weights to 1/N
    w = np.full(n_samples, (1 / n_samples))
    
    self.clfs = []
    # Iterate through classifiers
    for _ in range(self.n_clf):
        clf = DecisionStump()
        # Minimum error given for using a certain feature value threshold
        # for predicting sample label
        min_error = float('inf')
        # Iterate throught every unique feature value and see what value
        # makes the best threshold for predicting y
        for feature_i in range(n_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)
            # Try every unique feature value as threshold
            for threshold in unique_values:
                p = 1
                # Set all predictions to '1' initially
                prediction = np.ones(np.shape(y))
                # Label the samples whose values are below threshold as '-1'
                prediction[X[:, feature_i] < threshold] = -1
                # Error = sum of weights of misclassified samples
                error = sum(w[y != prediction])
                
                # If the error is over 50% we flip the polarity so that samples that
                # were classified as 0 are classified as 1, and vice versa
                # E.g error = 0.8 => (1 - error) = 0.2
                if error > 0.5:
                    error = 1 - error
                    p = -1

                # If this threshold resulted in the smallest error we save the
                # configuration
                if error < min_error:
                    clf.polarity = p
                    clf.threshold = threshold
                    clf.feature_index = feature_i
                    min_error = error
        # Calculate the alpha which is used to update the sample weights,
        # Alpha is also an approximation of this classifier's proficiency
        clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        # Set all predictions to '1' initially
        predictions = np.ones(np.shape(y))
        # The indexes where the sample values are below threshold
        negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
        # Label those as '-1'
        predictions[negative_idx] = -1
        # Calculate new weights 
        # Missclassified samples gets larger weights and correctly classified samples smaller
        w *= np.exp(-clf.alpha * y * predictions)
        # Normalize to one
        w /= np.sum(w)

        # Save classifier
        self.clfs.append(clf)
    return None

# Part 1b: Adaboost.predict
def predict(self, X):
    """Predict the class labels for the input samples.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input samples.

    Returns:
    --------
    y_pred : array-like, shape (n_samples,)
        The predicted class labels.
    """
    # Write your code here
    n_samples = np.shape(X)[0]
    y_pred = np.zeros((n_samples, 1))
    # For each classifier => label the samples
    for clf in self.clfs:
        # Set all predictions to '1' initially
        predictions = np.ones(np.shape(y_pred))
        # The indexes where the sample values are below threshold
        negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
        # Label those as '-1'
        predictions[negative_idx] = -1
        # Add predictions weighted by the classifiers alpha
        # (alpha indicative of classifier's proficiency)
        y_pred += clf.alpha * predictions

    # Return sign of prediction sum
    y_pred = np.sign(y_pred).flatten()

    return y_pred

# Part 2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the digits dataset
data = load_digits()

# Select digits 1 and 8
digit1 = 1
digit8 = 8

# Filter digits 1 and 8 from the dataset
X = data.data[(data.target == digit1) | (data.target == digit8)]
y = data.target[(data.target == digit1) | (data.target == digit8)]

# Transform labels to {-1, 1}
y[y == digit1] = -1
y[y == digit8] = 1

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create AdaBoost classifier with 50 weak learners
clf = AdaBoostClassifier(n_estimators=50)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.5f}")
print(f"Test Accuracy: {test_accuracy:.5f}")
