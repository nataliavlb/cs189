# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np
from sklearn import svm
import concurrent.futures

# 3a: Data Partitioning for MNIST
def mnist_data_partitioning():
    np.random.seed(42)
    indices = np.random.permutation(len(training_data_mnist))
    training_indices, validation_indices = indices[:55000], indices[55000:]
    
    X_train, X_val = training_data_mnist[training_indices], training_data_mnist[validation_indices]
    y_train, y_val = labels_mnist[training_indices], labels_mnist[validation_indices]
    
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    return X_train_flat, X_val_flat, y_train, y_val

# 3a: Data Partitioning for Spam

def spam_data_partitioning():
    np.random.seed(42)
    num_examples = len(training_data_spam)
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    split_index = int(0.8 * num_examples)
    X_train, X_val = training_data_spam[indices[:split_index]], training_data_spam[indices[split_index:]]
    y_train, y_val = labels_spam[indices[:split_index]], labels_spam[indices[split_index:]]

    return X_train, X_val, y_train, y_val

# 3b: Evaluation Metric (Accuracy)
def accuracy_score(y, y_pred):
    n = len(y)
    assert n == len(y_pred)
    sum = []
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            sum.append(1)
        else:
            sum.append(0)
    return (1/n) * np.sum(sum)

def results_to_csv(y_test, filename):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1
    df.to_csv(filename, index_label='Id')

# Load MNIST data
data_mnist = np.load(r'C:\Users\NataliaVillalobos\Downloads\cs189\hw1\data\mnist-data.npz')
training_data_mnist = data_mnist['training_data']
labels_mnist = data_mnist['training_labels']
test_data_mnist = data_mnist['test_data']

# Load spam data
data_spam = np.load(r'C:\Users\NataliaVillalobos\Downloads\cs189\hw1\data\spam-data.npz')
training_data_spam = data_spam['training_data']
labels_spam = data_spam['training_labels']
test_data_spam = data_spam['test_data']
c_values = [0.0000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]


# Partition and train MNIST data
X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist = mnist_data_partitioning()
clf_mnist = svm.SVC(kernel='linear', C=0.0001)
clf_mnist.fit(X_train_mnist, y_train_mnist)
mnist_test_predictions = clf_mnist.predict(test_data_mnist.reshape(len(test_data_mnist), -1))
results_to_csv(mnist_test_predictions, 'mnist_predictions.csv')

# Partition and train Spam data
X_train_spam, X_val_spam, y_train_spam, y_val_spam = spam_data_partitioning()
clf_spam = svm.SVC(kernel='linear', C=1000)
clf_spam.fit(X_train_spam, y_train_spam)
spam_test_predictions = clf_spam.predict(test_data_spam)
results_to_csv(spam_test_predictions, 'spam_predictions.csv')

print('Predictions saved to mnist_predictions.csv and spam_predictions.csv')

#accuracy for sanity 
mnist_train_accuracy = accuracy_score(y_train_mnist, clf_mnist.predict(X_train_mnist))
mnist_val_accuracy = accuracy_score(y_val_mnist, clf_mnist.predict(X_val_mnist))

spam_train_accuracy = accuracy_score(y_train_spam, clf_spam.predict(X_train_spam))
spam_val_accuracy = accuracy_score(y_val_spam, clf_spam.predict(X_val_spam))
print(f'MNIST Train Accuracy: {mnist_train_accuracy:.4f}, Validation Accuracy: {mnist_val_accuracy:.4f}')
print(f'Spam Train Accuracy: {spam_train_accuracy:.4f}, Validation Accuracy: {spam_val_accuracy:.4f}')

