"""
Have Fun!
- 189 Course Staff
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import mode
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from numpy import genfromtxt
import scipy.io
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io
import pandas as pd 
from sklearn.base import BaseEstimator, ClassifierMixin

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200, max_features='sqrt'):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.max_features = max_features
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, max_features=self.max_features, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        for tree in self.decision_trees:
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        if predictions.shape[0] > 2:
            from scipy.stats import mode
            final_prediction, _ = mode(predictions, axis=0)
            final_prediction = final_prediction.ravel()
        else: 
            final_prediction = np.sign(predictions.sum(axis=0))
            final_prediction[final_prediction == 0] = np.random.choice(predictions.ravel(), size=(final_prediction == 0).sum())
        return final_prediction


def fill_median(data, columns):
    for column in columns:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)
    return data

def one_hot_encode(data, columns):
    for column in columns:
        dummies = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, dummies], axis=1)
        data.drop(column, axis=1, inplace=True)
    return data

def preprocess(data):
    numeric_cols = ['age', 'sibsp', 'parch', 'fare']
    categorical_cols = ['sex', 'embarked', 'pclass']
    data = fill_median(data, numeric_cols)
    data = one_hot_encode(data, categorical_cols)

    return data

class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None
        self.split_idx, self.thresh = None, None
        self.pred = None

    def entropy(self, y):
        if len(y) == 0:
            return 0
        y_int = y.astype(int)
        counts = np.bincount(y_int)
        probabilities = counts / len(y_int)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


    def information_gain(self, X, y, idx, thresh):
        parent_entropy = self.entropy(y)
        left_indices = np.where(X[:, idx] <= thresh)[0]
        right_indices = np.where(X[:, idx] > thresh)[0]

        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])

        num_left, num_right = len(left_indices), len(right_indices)
        total = num_left + num_right
        weighted_entropy = (num_left / total) * left_entropy + (num_right / total) * right_entropy

        return parent_entropy - weighted_entropy

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def gini_purification(self, X, y, idx, thresh):
        left_indices = np.where(X[:, idx] < thresh)[0]
        right_indices = np.where(X[:, idx] >= thresh)[0]

        left_impurity = self.gini_impurity(y[left_indices])
        right_impurity = self.gini_impurity(y[right_indices])

        num_left, num_right = len(left_indices), len(right_indices)
        total = num_left + num_right

        weighted_impurity = (num_left / total) * left_impurity + (num_right / total) * right_impurity
        return weighted_impurity

    def fit(self, X, y, depth=0):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples <= 1 or len(np.unique(y)) == 1:
            self.pred = Counter(y).most_common(1)[0][0]
            return
        best_gain = -np.inf
        for idx in range(num_features):
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                gain = self.information_gain(X, y, idx, thresh)
                if gain > best_gain:
                    best_gain = gain
                    self.split_idx = idx
                    self.thresh = thresh
        if best_gain == -np.inf:
            self.pred = Counter(y).most_common(1)[0][0]
            return
        left_indices = np.where(X[:, self.split_idx] <= self.thresh)[0]
        right_indices = np.where(X[:, self.split_idx] > self.thresh)[0]
        self.left = DecisionTree(max_depth=self.max_depth - 1)
        self.right = DecisionTree(max_depth=self.max_depth - 1)
        self.left.fit(X[left_indices], y[left_indices])
        self.right.fit(X[right_indices], y[right_indices])


    def predict(self, X):
        if self.pred is not None:
            return np.array([self.pred] * X.shape[0])
        else:
            predictions = np.empty(X.shape[0], dtype=int)
            for i, row in enumerate(X):
                predictions[i] = self._predict_row(row)
            return predictions

    def _predict_row(self, row):
        if self.pred is not None:
            return self.pred
        if row[self.split_idx] <= self.thresh:
            return self.left._predict_row(row)
        else:
            return self.right._predict_row(row)

    def __repr__(self):
        if self.pred is not None:
            return f"Leaf: {self.pred}"
        else:
            return f"[{self.features[self.split_idx]} < {self.thresh}]\n Left: {self.left}\n Right: {self.right}"

    def print_tree(self, depth=0):
        indent = "  " * depth  # Indentation for the current level
        if self.pred is not None:
            return f"{indent}Leaf: {self.pred}"
        else:
            left_str = self.left.print_tree(depth + 1) if self.left else "None"
            right_str = self.right.print_tree(depth + 1) if self.right else "None"
            feature_label = self.features[self.split_idx] if self.features else f"Feature {self.split_idx}"
            return f"{indent}{feature_label} < {self.thresh}?\n{left_str}\n{right_str}"


spam_data = scipy.io.loadmat('datasets\\spam_data\\spam_data.mat')
print(spam_data.keys())
training_spam = spam_data['training_data']
test_spam = spam_data['test_data']
Y_spam = spam_data['training_labels'].ravel()
print(training_spam.shape, test_spam.shape, Y_spam.shape)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(training_spam, Y_spam)
predictions = dt.predict(test_spam)

submission_df = pd.DataFrame({'Id': np.arange(1, len(predictions) + 1), 'Category': predictions})
submission_df.to_csv('submission.csv', index=False)
print('CSV SUCCESS')
train_data = pd.read_csv('datasets/titanic/titanic_training.csv')
test_data = pd.read_csv('datasets/titanic/titanic_testing_data.csv')

# Preprocess the data
numeric_cols = ['age', 'sibsp', 'parch', 'fare']
categorical_cols = ['pclass', 'sex', 'embarked']

train_data = train_data.drop(['ticket', 'cabin'], axis=1)
train_data = fill_median(train_data, numeric_cols)
train_data = one_hot_encode(train_data, categorical_cols)

test_data = test_data.drop(['ticket', 'cabin'], axis=1)
test_data = fill_median(test_data, numeric_cols)
test_data = one_hot_encode(test_data, categorical_cols)
y_train = train_data['survived'].astype(int)
X_train = train_data.drop('survived', axis=1).to_numpy()

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
rf_model = DecisionTree(max_depth=3) 
rf_model.fit(X_train_split, y_train_split)

train_predictions = rf_model.predict(X_train_split)
val_predictions = rf_model.predict(X_val)
train_accuracy = accuracy_score(y_train_split, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

X_test = test_data.to_numpy() 
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

# Output the predictions
print(predictions)
submission_df = pd.DataFrame({'Id': np.arange(1, len(predictions) + 1), 'Category': predictions})
submission_df.to_csv('submission.csv', index=False)
print('CSV SUCCESS')






X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
depths = range(1, 41)
train_accuracies = []
val_accuracies = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    
    train_accuracies.append(accuracy_score(y_train, train_pred))
    val_accuracies.append(accuracy_score(y_val, val_pred))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy')
plt.plot(depths, val_accuracies, label='Validation Accuracy')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs. Accuracy')
plt.legend()
plt.show()

max_val_acc = max(val_accuracies)
optimal_depth = val_accuracies.index(max_val_acc) + 1 
print(f"The highest validation accuracy is {max_val_acc:.4f} at depth {optimal_depth}.")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

train_data = pd.read_csv('datasets/titanic/titanic_training.csv')
test_data = pd.read_csv('datasets/titanic/titanic_testing_data.csv')
X_train = train_data.drop('survived', axis=1)
y_train = train_data['survived']

categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

tree_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier(max_depth=3))])

tree_pipeline.fit(X_train, y_train)

decision_tree = tree_pipeline.named_steps['classifier']
plt.figure(figsize=(20,10))
plot_tree(decision_tree, filled=True)
plt.show()


















# train_data = pd.read_csv('datasets/titanic/titanic_training.csv')
# test_data = pd.read_csv('datasets/titanic/titanic_testing_data.csv')

# # Preprocess the data
# train_data = preprocess(train_data)
# test_data = preprocess(test_data)

# # Separate features and labels in the training data
# y_train = train_data['survived'].astype(int)  # Adjust the label column name based on your dataset
# X_train = train_data.drop('survived', axis=1)  # Adjust the label column name based on your dataset

# # Split the training data into a training and validation set
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# rf_model = RandomForest(n=200, max_features='sqrt')  # Adjust parameters as needed
# rf_model.fit(X_train_split, y_train_split)
# train_predictions = rf_model.predict(X_train_split)
# val_predictions = rf_model.predict(X_val)
# train_accuracy = accuracy_score(y_train_split, train_predictions)
# val_accuracy = accuracy_score(y_val, val_predictions)

# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Validation Accuracy: {val_accuracy:.4f}")
# rf_model.fit(X_train, y_train)
# predictions = rf_model.predict(test_data)  





# train_data = pd.read_csv('datasets/titanic/titanic_training.csv')
# test_data = pd.read_csv('datasets/titanic/titanic_testing_data.csv')
# train_data = train_data.drop(['ticket','cabin'], axis=1)
# test_data = test_data.drop(['ticket','cabin'], axis=1)
# numeric_cols = ['age', 'sibsp', 'parch', 'fare']
# categorical_cols = ['pclass','sex', 'embarked']
# train_data = fill_median(train_data, numeric_cols)
# train_data = one_hot_encode(train_data, categorical_cols)
# print(train_data)
# y_train = train_data['survived'].astype(int)
# X_train = train_data.drop(['survived'], axis=1)
# X_train = X_train.to_numpy()
# test_data = fill_median(test_data, numeric_cols)
# test_data = one_hot_encode(test_data, categorical_cols)
# X_test = test_data.to_numpy()
# print(X_train)



# rf_model = DecisionTree(max_depth=3)
# y_train = y_train.to_numpy()
# rf_model.fit(X_train, y_train)
# predictions = rf_model.predict(X_test)
# print(predictions)
# submission_df = pd.DataFrame({'Id': np.arange(1, len(predictions) + 1), 'Category': predictions})
# submission_df.to_csv('submission.csv', index=False)
# print('CSV SUCCESS')

# X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# rf_model = DecisionTree(max_depth=3)  
# rf_model.fit(X_train_full, y_train_full)
# train_predictions = rf_model.predict(X_train_full)
# val_predictions = rf_model.predict(X_val)
# train_accuracy = accuracy_score(y_train_full, train_predictions)
# val_accuracy = accuracy_score(y_val, val_predictions)
# print(f"Training Accuracy: {train_accuracy}")
# print(f"Validation Accuracy: {val_accuracy}")



# train_data = fill_median(train_data, numeric_cols)
# train_data = one_hot_encode(train_data, categorical_cols)
# print(train_data)
# y_train = train_data['survived'].astype(int)
# X_train = train_data.drop(['survived'], axis=1)
# X_train = X_train.to_numpy()
# test_data = fill_median(test_data, numeric_cols)
# test_data = one_hot_encode(test_data, categorical_cols)
# X_test = test_data.to_numpy()
# print(X_train)
# rf_model = DecisionTree(max_depth=3)
# y_train = y_train.to_numpy()
# rf_model.fit(X_train, y_train)
# predictions = rf_model.predict(X_test)
# print(predictions)
# submission_df = pd.DataFrame({'Id': np.arange(1, len(predictions) + 1), 'Category': predictions})
# submission_df.to_csv('submission.csv', index=False)
# print('CSV SUCCESS')













# class BoostedRandomForest(RandomForest):

#     def fit(self, X, y):
#         # TODO
#         pass
    
#     def predict(self, X):
#         # TODO
#         pass


# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder




# def evaluate(clf):
#     print("Cross validation", cross_val_score(clf, X, y))
#     if hasattr(clf, "decision_trees"):
#         counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
#         first_splits = [
#             (features[term[0]], term[1]) for term in counter.most_common()
#         ]
#         print("First splits", first_splits)


# if __name__ == "__main__":
#     # dataset = "titanic" or "spam"
#     if dataset == "titanic":
#         # Assuming you have loaded the Titanic data into 'data' variable as a DataFrame
#         X, feature_labels = preprocess(data.drop('survived', axis=1), onehot_cols=['sex', 'embarked', 'pclass'])
#         y = data['survived']

#         # Now initialize and train your custom DecisionTree or sklearn's DecisionTreeClassifier
#         tree = DecisionTree(max_depth=3, feature_labels=feature_labels)
#         tree.fit(X, y)

#     elif dataset == "spam":
#         # Spam data loading and processing
#         X, y = training_spam, Y_spam
#         feature_labels = [f'Feature {i+1}' for i in range(X.shape[1])]

#         tree = DecisionTree(max_depth=3, feature_labels=feature_labels)
#         tree.fit(X, y)

#     # Output part, predictions, and evaluations
#     print(tree.print_tree())

#     N = 100

#     if dataset == "titanic":
#         # Load titanic data
#         path_train = 'datasets/titanic/titanic_training.csv'
#         data = genfromtxt(path_train, delimiter=',', dtype=None, encoding='utf-8')
#         path_test = 'datasets/titanic/titanic_testing_data.csv'
#         test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding='utf-8')
#         y = data[1:, 0]  # label = survived
#         class_names = ["Died", "Survived"]

#         labeled_idx = np.where(y != b'')[0]
#         y = np.array(y[labeled_idx], dtype=float).astype(int)
#         print("\n\nPart (b): preprocessing the titanic dataset")
#         X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
#         X = X[labeled_idx, :]
#         Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
#         assert X.shape[1] == Z.shape[1]
#         features = list(data[0, 1:]) + onehot_features

#     elif dataset == "spam":
#         features = [
#             "pain", "private", "bank", "money", "drug", "spam", "prescription",
#             "creative", "height", "featured", "differ", "width", "other",
#             "energy", "business", "message", "volumes", "revision", "path",
#             "meter", "memo", "planning", "pleased", "record", "out",
#             "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
#             "square_bracket", "ampersand"
#         ]
#         assert len(features) == 32

#         # Load spam data
#         path_train = 'datasets/spam_data/spam_data.mat'
#         data = scipy.io.loadmat(path_train)
#         X = data['training_data']
#         y = np.squeeze(data['training_labels'])
#         Z = data['test_data']
#         class_names = ["Ham", "Spam"]

#     else:
#         raise NotImplementedError("Dataset %s not handled" % dataset)

#     print("Features", features)
#     print("Train/test size", X.shape, Z.shape)
    
#     print("\n\nPart 0: constant classifier")
#     print("Accuracy", 1 - np.sum(y) / y.size)

#     # sklearn decision tree
#     print("\n\nsklearn's decision tree")
#     clf = DecisionTreeClassifier(random_state=0, **params)
#     clf.fit(X, y)
#     evaluate(clf)
#     out = io.StringIO()
#     export_graphviz(
#         clf, out_file=out, feature_names=features, class_names=class_names)
#     # For OSX, may need the following for dot: brew install gprof2dot
#     graph = graph_from_dot_data(out.getvalue())
#     graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    
#     # TODO







