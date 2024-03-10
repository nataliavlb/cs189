import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def normalize_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data_normalized = (train_data - mean) / std
    test_data_normalized = (test_data - mean) / std
    return train_data_normalized, test_data_normalized

def add_fictitious_dimension(data):
    return np.c_[np.ones(data.shape[0]), data]

def sigmoid(z):
    return expit(z)

def compute_cost(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    regularization_term = (lambda_ / (2 * m)) * np.sum(theta[1:]**2)
    cost = -(1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h))) + regularization_term
    return cost

def batch_gradient_descent(X, y, theta, alpha, lambda_, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        regularization_term = (lambda_ / m) * theta
        regularization_term[0] = 0  
        theta = theta - alpha * (1/m) * (np.dot(X.T, (h - y)) + regularization_term)
       
        cost = compute_cost(X, y, theta, lambda_)
        cost_history.append(cost)
        
    return theta, cost_history

data = loadmat('data2024.mat')
y = data['y'].flatten()
X = data['X']
X_test = data['X_test']
description = data['description']

X_normalized, X_test_normalized = normalize_data(X, X_test)

X_normalized = add_fictitious_dimension(X_normalized)
X_test_normalized = add_fictitious_dimension(X_test_normalized)


X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

num_features = X_train.shape[1]
theta_initial = np.zeros(num_features)

delta = 0.1
alpha = 0.01  
lambdaa = 0.1 
num_iterations = 1000

theta_final, cost_history = batch_gradient_descent(X_train, y_train, theta_initial, alpha, lambdaa, num_iterations)

plt.plot(range(1, num_iterations + 1), cost_history, marker='o')
plt.title('Cost Function vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()

print("Learned Parameters:", theta_final)





def stochastic_gradient_descent(X, y, theta, alpha, lambda_, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            h = sigmoid(np.dot(xi, theta))
            regularization_term = (lambda_ / m) * theta
            regularization_term[0] = 0  
            theta = theta - alpha * (np.dot(xi.T, (h - yi)) + regularization_term)
    
        cost = compute_cost(X, y, theta, lambda_)
        cost_history.append(cost)
        
    return theta, cost_history


def stochastic_gradient_descent_variable_step(X, y, theta, delta, lambda_, num_iterations):
    m = len(y)
    cost_history = []

    for t in range(1, num_iterations + 1):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]

        epsilon_t = delta / t

        h = sigmoid(np.dot(xi, theta))
        regularization_term = (lambda_ / m) * theta
        regularization_term[0] = 0  # Exclude bias term from regularization
        theta = theta - epsilon_t * (np.dot(xi.T, (h - yi)) + regularization_term)

        # Compute and store the cost for monitoring
        cost = compute_cost(X, y, theta, lambda_)
        cost_history.append(cost)

    return theta, cost_history

theta_final, cost_history = stochastic_gradient_descent(X_train, y_train, theta_initial, alpha, lambdaa, num_iterations)

plt.plot(range(1, num_iterations + 1), cost_history, marker='o')
plt.title('Cost Function vs. Iterations (Stochastic Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()

print("Learned Parameters:", theta_final)



theta_variable_step, cost_history_variable_step = stochastic_gradient_descent_variable_step(
    X_train, y_train, theta_initial, delta, lambdaa, num_iterations
)


plt.plot(range(1, num_iterations + 1), cost_history_variable_step, label='Variable Step Size')
plt.title('Cost Function vs. Iterations (SGD with Time-Varying Step Size)')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.legend()
plt.show()

print("Learned Parameters (Variable Step Size):", theta_variable_step)



theta_final, _ = batch_gradient_descent(X_normalized, y, theta_initial, alpha, lambdaa, num_iterations)

test_predictions = sigmoid(np.dot(X_test_normalized, theta_final))

binary_predictions = (test_predictions >= 0.5).astype(int)

submission_df = pd.DataFrame({'Id': np.arange(1, len(binary_predictions) + 1), 'Category': binary_predictions})

submission_df.to_csv('submission.csv', index=False)
print('CSV SUCCESS')