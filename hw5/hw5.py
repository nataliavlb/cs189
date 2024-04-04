import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
spam_data = scipy.io.loadmat('datasets\\spam_data\\spam_data.mat')
print(spam_data.keys())
training_spam = spam_data['training_data']
test_spam = spam_data['test_data']
Y_spam = spam_data['training_labels'].ravel()
print(training_spam.shape, test_spam.shape, Y_spam.shape)

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return DecisionTreeNode(value=self._majority_class(y))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return DecisionTreeNode(value=self._majority_class(y))

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(feature_index, threshold, left_child, right_child)

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -np.inf
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _entropy(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _information_gain(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        y_left, y_right = y[left_indices], y[right_indices]

        parent_entropy = self._entropy(y)
        num_left, num_right = len(y_left), len(y_right)

        if num_left == 0 or num_right == 0:
            return 0

        weighted_entropy = (num_left / (num_left + num_right)) * self._entropy(y_left) + \
                           (num_right / (num_left + num_right)) * self._entropy(y_right)
        info_gain = parent_entropy - weighted_entropy
        return info_gain

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Example usage:
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(training_spam, Y_spam)
predictions = dt.predict(test_spam)

submission_df = pd.DataFrame({'Id': np.arange(1, len(predictions) + 1), 'Category': predictions})
submission_df.to_csv('submission.csv', index=False)
print('CSV SUCCESS')