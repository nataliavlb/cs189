# import numpy as np
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# import os
# import pandas as pd

# data = np.load(r'C:\Users\NataliaVillalobos\Downloads\cs189\hw1\data\mnist-data.npz')
# training_data = data['training_data'] 
# Y = data['training_labels'] 
# n = len(training_data)
# training_examples = [100, 200, 500, 1000, 2000, 5000, 10000] 
# c_values= [0.0000001,0.0001,0.001,0.01, 0.1, 1, 10, 100, 1000]

# def accuracy_score(y, y_pred):
#     n = len(y)
#     assert n == len(y_pred)
#     sum = []
#     for i in range(len(y)):
#         if y[i] == y_pred[i]:
#             sum.append(1)
#         else:
#             sum.append(0)
#     return (1/n) * np.sum(sum)

# def train_mnist(training_data, Y, n, training_examples):
#     np.random.seed(42)
#     shuffle_indices = np.random.permutation(n)
#     shuffled_data = training_data[shuffle_indices]
#     shuffled_labels = Y[shuffle_indices]
#     validation_data = shuffled_data[:10000]
#     validation_labels = shuffled_labels[:10000]
#     #print(validation_data, validation_labels)

#     for size in training_examples:
#         train_data_subset = shuffled_data[:size]
#         train_labels_subset = shuffled_labels[:size]
#         train_data_flat = train_data_subset.reshape(size, -1)
#         validation_data_flat = validation_data.reshape(10000, -1)

#         clf = svm.SVC(kernel='linear')
#         clf.fit(train_data_flat, train_labels_subset)
#         validation_predictions = clf.predict(validation_data_flat)
#         accuracy = accuracy_score(validation_labels, validation_predictions)
       
#         # print("MNIST Accuracy")
#         #print(f"Training size: {size}, Validation Accuracy: {accuracy * 100:.2f}%")
#         best_accuracy = 0
#         best_c = None

#         for c_value in c_values:
#             clf = svm.SVC(kernel='linear', C=c_value)
#             clf.fit(train_data_flat, train_labels_subset)
#             validation_predictions = clf.predict(validation_data_flat)
#             accuracy = accuracy_score(validation_labels, validation_predictions)

#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_c = c_value

#         print(f"Training size: {size}, Best C: {best_c}, Validation Accuracy: {best_accuracy * 100:.2f}%")
# print("Question 3a and 4a: MNIST partitioning and accuracy")
# #train_mnist(training_data, Y, n, training_examples)


# spam_data = np.load(r'C:\Users\NataliaVillalobos\Downloads\cs189\hw1\data\spam-data.npz')
# X = spam_data['training_data']
# Y = spam_data['training_labels']


# def train_spam(X,Y,training_examples):

#     X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#     for size in training_examples:
#         X_train_subset = X_train[:size]
#         Y_train_subset = Y_train[:size]
#         clf = svm.SVC(kernel='linear')
#         clf.fit(X_train_subset, Y_train_subset)
#         predictions_val = clf.predict(X_val)
#         accuracy_val = accuracy_score(Y_val, predictions_val)
#         print(f"Training size: {size}, Validation Accuracy: {accuracy_val * 100:.2f}%")

# print("Question 3a and 4b: Spam partitioning and accuracy")
# #train_spam(X,Y,training_examples)
# def k_fold_cross_validation_spam(X, Y, c_values, k=5):
#     np.random.seed(42)
#     n = min(len(X), len(Y))
#     shuffle_indices = np.random.permutation(n)
#     shuffled_data = X[shuffle_indices]
#     shuffled_labels = Y[shuffle_indices]
#     fold_size = n // k
#     best_accuracy = 0
#     best_c = None

#     for c_value in c_values:
#         total_accuracy = 0
#         for fold in range(k):
#             validation_start = fold * fold_size
#             validation_end = (fold + 1) * fold_size
#             validation_data = shuffled_data[validation_start:validation_end].reshape(-1, 784)
#             validation_labels = shuffled_labels[validation_start:validation_end]
#             train_data = np.concatenate([shuffled_data[:validation_start], shuffled_data[validation_end:]]).reshape(-1, 784)
#             train_labels = np.concatenate([shuffled_labels[:validation_start], shuffled_labels[validation_end:]])


#             clf = svm.SVC(kernel='linear', C=c_value)
#             clf.fit(train_data, train_labels)

#             validation_predictions = clf.predict(validation_data)
#             accuracy = accuracy_score(validation_labels, validation_predictions)
#             total_accuracy += accuracy

#         average_accuracy = total_accuracy / k

#         if average_accuracy > best_accuracy:
#             best_accuracy = average_accuracy
#             best_c = c_value

#         print(f"C: {c_value}, Average Cross-Validation Accuracy: {average_accuracy * 100:.2f}%")

#     print(f"Best C: {best_c}, Best Cross-Validation Accuracy: {best_accuracy * 100:.2f}%")

# # Run cross-validation
# print("5-Fold Cross-Validation for Spam Dataset")
# #k_fold_cross_validation_spam(training_data, Y, c_values)


# best_c_mnist = 0.0001 
# clf_mnist = svm.SVC(kernel='linear', C=best_c_mnist)
# clf_mnist.fit(training_data.reshape(n, -1), Y)
# mnist_test_predictions = clf_mnist.predict(data['test_data'].reshape(len(data['test_data']), -1))

# # Spam
# best_c_spam = 100
# clf_spam = svm.SVC(kernel='linear', C=best_c_spam)
# clf_spam.fit(X, Y)
# spam_test_predictions = clf_spam.predict(spam_data['test_data'])

# mnist_submission_df = pd.DataFrame({'Id': np.arange(1, len(mnist_test_predictions)+1), 'Category': mnist_test_predictions})
# spam_submission_df = pd.DataFrame({'Id': np.arange(1, len(spam_test_predictions)+1), 'Category': spam_test_predictions})

# mnist_submission_df.to_csv('mnist_predictions.csv', index=False)
# spam_submission_df.to_csv('spam_predictions.csv', index=False)



