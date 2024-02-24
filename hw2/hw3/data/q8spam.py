import pandas as pd
import numpy as np 
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt

data_spam = np.load('spam-data-hw3.npz')
training_data = data_spam['training_data']
training_labels = data_spam['training_labels']
test_data = data_spam['test_data']

def compute_class_means(X, y):
    class_means = []
    for c in np.unique(y):
        class_means.append(np.mean(X[y == c], axis=0))
    return np.array(class_means)

# Function to compute the pooled covariance matrix
def compute_pooled_covariance(X, y):
    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    pooled_covariance = np.zeros((num_features, num_features))

    for c in np.unique(y):
        class_samples = X[y == c]
        class_covariance = np.cov(class_samples, rowvar=False)
        class_size = len(class_samples)
        pooled_covariance += (class_size - 1) / (len(X) - num_classes) * class_covariance

    return pooled_covariance

def compute_class_priors(y):
    class_priors = [np.mean(y == c) for c in np.unique(y)]
    return np.array(class_priors)

def lda_classify(X, class_means, pooled_covariance, priors):
    inv_covariance = inv(pooled_covariance)
    discriminant = X @ inv_covariance @ class_means.T - 0.5 * np.sum((class_means @ inv_covariance) * class_means, axis=1)
    discriminant += np.log(priors)
    predictions = np.argmax(discriminant, axis=1)
    return predictions

np.random.seed(42)
indices = np.random.permutation(len(training_data))
training_indices, validation_indices = indices[:4000], indices[4000:]
X_train, X_val = training_data[training_indices], training_data[validation_indices]
y_train, y_val = training_labels[training_indices], training_labels[validation_indices]
class_means = compute_class_means(X_train, y_train)
pooled_covariance = compute_pooled_covariance(X_train, y_train)
class_priors = compute_class_priors(y_train)
lda_predictions = lda_classify(X_val, class_means, pooled_covariance, class_priors)
accuracy = np.mean(lda_predictions == y_val)
print(f"Validation Accuracy: {accuracy}")
test_predictions = lda_classify(test_data, class_means, pooled_covariance, class_priors)

lda_predictions_val = lda_classify(X_val, class_means, pooled_covariance, class_priors)

lda_predictions_test = lda_classify(test_data, class_means, pooled_covariance, class_priors)
accuracy_val = np.mean(lda_predictions_val == y_val)
print(f"Validation Accuracy: {accuracy_val}")

def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission.csv', index_label='Id')

results_to_csv(test_predictions)
