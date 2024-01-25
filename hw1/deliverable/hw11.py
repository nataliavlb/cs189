import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# 4a: Train Linear SVM on MNIST
def train_linear_svm_mnist(X_train, X_val, y_train, y_val, num_training_examples):
    c_values = [0.0000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    for size in num_training_examples:
        train_data_subset, train_labels_subset = X_train[:size], y_train[:size]
        val_data_flat = X_val.reshape(len(X_val), -1)

        # Train linear SVM
        clf = svm.SVC(kernel='linear',c=c_values[size])
        clf.fit(train_data_subset.reshape(size, -1), train_labels_subset)
        val_predictions = clf.predict(val_data_flat)
        acc = accuracy_score(y_val, val_predictions)
       
        # Hyperparameter tuning
        best_accuracy = 0
        best_c = None
        for c_value in c_values:
            clf = svm.SVC(kernel='linear', C=c_value)
            clf.fit(train_data_subset.reshape(size, -1), train_labels_subset)
            val_predictions = clf.predict(val_data_flat)
            acc = accuracy_score(y_val, val_predictions)

            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c_value

        print(f"Training size: {size}, Best C: {best_c}, Validation Accuracy: {best_accuracy * 100:.2f}%")

# 4b: Train Linear SVM on Spam
def train_linear_svm_spam(X_train, X_val, y_train, y_val, num_training_examples):
    c_values = [0.0000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    for size in num_training_examples:
        X_train_subset, y_train_subset = X_train[:size], y_train[:size]
        
        # Train linear SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_subset, y_train_subset)
        predictions_val = clf.predict(X_val)
        accuracy_val = accuracy_score(y_val, predictions_val)
        print(f"Training size: {size}, Validation Accuracy: {accuracy_val * 100:.2f}%")

# 5: Hyperparameter Tuning for MNIST
def hyperparameter_tuning_mnist(X_train, X_val, y_train, y_val):
    c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_accuracy = 0
    best_c = None
    
    for c_value in c_values:
        clf = svm.SVC(kernel='linear', C=c_value)
        clf.fit(X_train.reshape(len(X_train), -1), y_train)
        val_predictions = clf.predict(X_val.reshape(len(X_val), -1))
        acc = accuracy_score(y_val_mnist, val_predictions)

        if acc > best_accuracy:
            best_accuracy = acc
            best_c = c_value

        print(f"C: {c_value}, Validation Accuracy: {acc * 100:.2f}%")

    print(f"Best C for MNIST: {best_c}, Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# 6: K-Fold Cross-Validation for Spam
def k_fold_cross_validation_spam(X, y, c_values, k=5):
    np.random.seed(42)
    n = min(len(X), len(y))
    shuffle_indices = np.random.permutation(n)
    shuffled_data = X[shuffle_indices]
    shuffled_labels = y[shuffle_indices]
    fold_size = n // k
    best_accuracy = 0
    best_c = None

    for fold in range(k):

        validation_start = fold * fold_size
        validation_end = (fold + 1) * fold_size
        validation_data = shuffled_data[validation_start:validation_end]
        validation_labels = shuffled_labels[validation_start:validation_end]
        train_data = np.concatenate([shuffled_data[:validation_start], shuffled_data[validation_end:]])
        train_labels = np.concatenate([shuffled_labels[:validation_start], shuffled_labels[validation_end:]])


        clf = svm.SVC(kernel='linear', C=1000)  
        clf.fit(train_data, train_labels)

        val_predictions = clf.predict(validation_data)

        accuracy_val = accuracy_score(validation_labels, val_predictions)
        print(f"Fold {fold + 1}: Validation Accuracy: {accuracy_val * 100:.2f}%")

        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_c = clf.C

    print(f"Best C for Spam: {best_c}, Best Validation Accuracy: {best_accuracy * 100:.2f}%")

            

X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist = mnist_data_partitioning()
X_train_spam, X_val_spam, y_train_spam, y_val_spam = spam_data_partitioning()
train_linear_svm_mnist(X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist, [100, 200, 500, 1000, 2000, 5000, 10000])
train_linear_svm_spam(X_train_spam, X_val_spam, y_train_spam, y_val_spam, [100, 200, 500, 1000, 2000])
hyperparameter_tuning_mnist(X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist)
k_fold_cross_validation_spam(X_train_spam, y_train_spam, c_values)


clf_spam = svm.SVC(kernel='linear')
clf_mnist = svm.SVC(kernel='linear')



# #Sanity check/ dims and such 
# print("MNIST Training Data:")
# print(training_data_mnist[:5])  # Print the first 5 samples

# print("\nSpam Training Data:")
# print(training_data_spam[:5])  # Print the first 5 samples

# print("MNIST SVM Model Parameters:")
# print(clf_mnist.get_params())

# print("\nSpam SVM Model Parameters:")
# print(clf_spam.get_params())

# print("MNIST Test Data:")
# print(test_data_mnist[:5])  # Print the first 5 samples

# print("\nSpam Test Data:")
# print(test_data_spam[:5])  # Print the first 5 samples

















# clf_mnist = svm.SVC(kernel='linear', C=0.001)
# clf_mnist.fit(training_data_mnist.reshape(len(training_data_mnist), -1), labels_mnist)
# mnist_test_predictions = clf_mnist.predict(test_data_mnist.reshape(len(test_data_mnist), -1))

# mnist_submission_df = pd.DataFrame({'Id': np.arange(1, len(mnist_test_predictions)+1), 'Category': mnist_test_predictions})
# mnist_submission_df.to_csv('mnist_predictions.csv', index=False)

# clf_spam = svm.SVC(kernel='linear', C=1e-07)
# clf_spam.fit(training_data_spam, labels_spam)
# spam_test_predictions = clf_spam.predict(test_data_spam)

# spam_submission_df = pd.DataFrame({'Id': np.arange(1, len(spam_test_predictions)+1), 'Category': spam_test_predictions})
# spam_submission_df.to_csv('spam_predictions.csv', index=False)
# print('worked')

