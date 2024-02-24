import pandas as pd
import numpy as np 
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.linalg import inv


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission.csv', index_label='Id')


#Question 1 
#load dataset
mnist = np.load('mnist-data-hw3.npz')
print("Keys in the .npz file:", list(mnist.keys()))
for key in mnist.keys():
    print(f"{key}:")
    print(mnist[key])

training_data=mnist['training_data']
print(training_data.shape)
training_labels=mnist['training_labels']
print(training_labels.shape)
test_data= mnist['test_data']
print(test_data.shape)

def fit_gaussian_naive_bayes(X, y):
    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    means = np.zeros((num_classes, num_features))
    variances = np.zeros((num_classes, num_features))
    class_probs = np.zeros(num_classes)
    for i in range(num_classes):
        class_indices = (y == i)
        class_probs[i] = np.mean(class_indices)
        means[i] = np.mean(X[class_indices], axis=0)
        variances[i] = np.var(X[class_indices], axis=0)
    return class_probs, means, variances

def predict_gaussian_naive_bayes(X, class_probs, means, variances):
    num_classes = len(class_probs)
    num_samples = X.shape[0]

    log_likelihoods = np.zeros((num_samples, num_classes))

    for i in range(num_classes):
        cov_matrix = np.diag(variances[i]) + 1e-6 * np.eye(len(variances[i]))
        log_likelihoods[:, i] = multivariate_normal.logpdf(X, mean=means[i], cov=cov_matrix)
    log_probs = log_likelihoods + np.log(class_probs)
    log_sum_probs = logsumexp(log_probs, axis=1)
    log_normalized_probs = log_probs - log_sum_probs[:, np.newaxis]
    predictions = np.argmax(log_normalized_probs, axis=1)

    return predictions

num_samples, _, height, width = training_data.shape
flattened_data = training_data.reshape(num_samples, -1)
normalized_data = flattened_data / np.linalg.norm(flattened_data, axis=1)[:, np.newaxis]
split_ratio = 0.8
split_index = int(split_ratio * num_samples)
X_train, X_val = normalized_data[:split_index], normalized_data[split_index:]
y_train, y_val = training_labels[:split_index], training_labels[split_index:]
class_probs, means, variances = fit_gaussian_naive_bayes(X_train, y_train)
predictions = predict_gaussian_naive_bayes(X_val, class_probs, means, variances)
accuracy = np.mean(predictions == y_val)
print("Accuracy:", accuracy)
results_to_csv(predictions[:10000])

#Question 2

digit_class = 2
cov_matrix = np.diag(variances[digit_class]) + 1e-6 * np.eye(len(variances[digit_class]))

plt.figure(figsize=(8, 8))
plt.imshow(cov_matrix, cmap='viridis', interpolation='none')
plt.title(f'Covariance Matrix for Digit {digit_class}')
plt.colorbar()
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.show()


#Question 3a
def compute_pooled_covariance(X, y):
    num_classes = len(np.unique(y))
    num_features = X.shape[1]

    pooled_covariance = np.zeros((num_features, num_features))

    class_covariances = [np.cov(X[y == c], rowvar=False) for c in range(num_classes)]
    class_counts = np.array([np.sum(y == c) for c in range(num_classes)])
    
    total_samples = np.sum(class_counts)

    for c in range(num_classes):
        weight = class_counts[c] / total_samples
        pooled_covariance += weight * class_covariances[c]

    return pooled_covariance

def lda_classify(X, means, pooled_covariance, priors):
    discriminant = X @ np.linalg.pinv(pooled_covariance) @ means.T - 0.5 * np.sum((means @ np.linalg.pinv(pooled_covariance)) * means, axis=1)

    discriminant += np.log(priors)
    predictions = np.argmax(discriminant, axis=1)

    return predictions


np.random.seed(42)
validation_indices = np.random.choice(len(X_train), size=10000, replace=False)
X_val = X_train[validation_indices]
y_val = y_train[validation_indices]

training_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]


lda_error_rates = []

for num_points in training_points:

    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    class_means = np.array([np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)])


    pooled_covariance = compute_pooled_covariance(X_train_subset, y_train_subset)
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])

    #LDA
    lda_predictions = lda_classify(X_val, class_means, pooled_covariance, priors)
    error_rate = 1 - np.mean(lda_predictions == y_val)
    lda_error_rates.append(error_rate)

def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission.csv', index_label='Id')


plt.plot(training_points, lda_error_rates, marker='o', label='LDA')
plt.title('Error Rate vs Number of Training Points (LDA)')
plt.xlabel('Number of Training Points')
plt.ylabel('Error Rate')
plt.legend()
plt.show()



#Question 3b

def qda_classify(X, means, covariances, priors, epsilon=1e-5):
    discriminant = np.zeros((len(X), 10))

    for c in range(10):
        try:
            covariance_inv = np.linalg.pinv(covariances[c])
        except np.linalg.LinAlgError:
            covariance_inv = np.linalg.pinv(covariances[c] + epsilon * np.eye(covariances[c].shape[0]))

        X_centered = X - means[c]
        discriminant[:, c] = -0.5 * np.sum(X_centered @ covariance_inv * X_centered, axis=1)

        sign, logdet = np.linalg.slogdet(covariances[c] + epsilon * np.eye(covariances[c].shape[0]))
        discriminant[:, c] += np.log(priors[c]) - 0.5 * logdet

    predictions = np.argmax(discriminant, axis=1)

    return predictions

qda_error_rates = []

# Evaluate error rates for QDA with different numbers of training points
for num_points in training_points:
    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    class_means = [np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)]
    class_covariances = [np.cov(X_train_subset[y_train_subset == c], rowvar=False) for c in range(10)]
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])
    qda_predictions = qda_classify(X_val, class_means, class_covariances, priors)

    error_rate = 1 - np.mean(qda_predictions == y_val)
    qda_error_rates.append(error_rate)

plt.plot(training_points, qda_error_rates, marker='o', label='QDA')
plt.title('Error Rate vs Number of Training Points (QDA)')
plt.xlabel('Number of Training Points')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

#Question 3d 
qda_error_rates_per_digit = [[] for _ in range(10)]
for num_points in training_points:
    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    class_means = [np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)]
    class_covariances = [np.cov(X_train_subset[y_train_subset == c], rowvar=False) for c in range(10)]
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])
    qda_predictions = qda_classify(X_val, class_means, class_covariances, priors)
    for digit in range(10):
        error_rate = 1 - np.mean(qda_predictions[y_val == digit] == digit)
        qda_error_rates_per_digit[digit].append(error_rate)

plt.figure(figsize=(10, 6))
for digit in range(10):
    plt.plot(training_points, qda_error_rates_per_digit[digit], marker='o', label=f'Digit {digit}')

plt.title('Validation Error vs Number of Training Points (QDA) for Each Digit')
plt.xlabel('Number of Training Points')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

qda_accuracies_per_digit = [[] for _ in range(10)]

for num_points in training_points:
    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    class_means = [np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)]
    class_covariances = [np.cov(X_train_subset[y_train_subset == c], rowvar=False) for c in range(10)]
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])
    
    qda_predictions = qda_classify(X_val, class_means, class_covariances, priors)
    
    for digit in range(10):
        accuracy = np.mean(qda_predictions[y_val == digit] == digit)
        qda_accuracies_per_digit[digit].append(accuracy)

plt.figure(figsize=(10, 6))
for digit in range(10):
    plt.plot(training_points, qda_accuracies_per_digit[digit], marker='o', label=f'Digit {digit}')

plt.title('Validation Accuracy vs Number of Training Points (QDA) for Each Digit')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


qda_accuracies_per_digit = [[] for _ in range(10)]

for num_points in training_points:
    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    class_means = [np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)]
    class_covariances = [np.cov(X_train_subset[y_train_subset == c], rowvar=False) for c in range(10)]
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])
    
    qda_predictions = qda_classify(X_val, class_means, class_covariances, priors)
    
    for digit in range(10):
        accuracy = np.mean(qda_predictions[y_val == digit] == digit)
        qda_accuracies_per_digit[digit].append(accuracy)

plt.figure(figsize=(10, 6))
for digit in range(10):
    plt.plot(training_points, qda_accuracies_per_digit[digit], marker='o', label=f'Digit {digit}')

plt.title('Validation Accuracy vs Number of Training Points (QDA) for Each Digit')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



lda_accuracies_per_digit = [[] for _ in range(10)]

for num_points in training_points:
    train_indices = np.random.choice(len(X_train), size=num_points, replace=True)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    class_means = [np.mean(X_train_subset[y_train_subset == c], axis=0) for c in range(10)]
    class_covariances = [np.cov(X_train_subset[y_train_subset == c], rowvar=False) for c in range(10)]
    priors = np.array([np.mean(y_train_subset == c) for c in range(10)])
    
    qda_predictions = lda_classify(X_val, class_means, class_covariances, priors)
    
    for digit in range(10):
        accuracy = np.mean(lda_predictions[y_val == digit] == digit)
        lda_error_rates[digit].append(accuracy)

plt.figure(figsize=(10, 6))
for digit in range(10):
    plt.plot(training_points, qda_accuracies_per_digit[digit], marker='o', label=f'Digit {digit}')

plt.title('Validation Accuracy vs Number of Training Points (LDA) for Each Digit')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

data_spam = np.load('spam-data-hw3.npz')
training_data = data_spam['training_data']
training_labels = data_spam['training_labels']
test_data = data_spam['test_data']

def compute_class_means(X, y):
    class_means = []
    for c in np.unique(y):
        class_means.append(np.mean(X[y == c], axis=0))
    return np.array(class_means)


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

