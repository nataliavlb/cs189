import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random number seed for reproducibility
np.random.seed(42)

# Function to generate random samples
def generate_samples(n):
    X1 = np.random.normal(loc=3, scale=3, size=n)
    X2 = 0.5 * X1 + np.random.normal(loc=4, scale=2, size=n)
    samples = np.column_stack((X1, X2))
    return samples

# Generate 100 random samples
n = 100
samples = generate_samples(n)

#1: Compute the mean of the sample in R^2
mean_sample = np.mean(samples, axis=0)
print("Mean of the sample in R^2:", mean_sample)

#2: Compute the 2 × 2 covariance matrix of the sample based on the sample mean
covariance_matrix = np.cov(samples, rowvar=False)
print("Covariance Matrix of the sample:\n", covariance_matrix)

#3: Compute the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

#4:
def plot_data_and_eigenvectors(samples, mean, eigenvectors, eigenvalues):
    plt.figure(figsize=(8, 8))  # Set the figure size explicitly
    plt.scatter(samples[:, 0], samples[:, 1], label='Data Points')
    plt.quiver(mean[0], mean[1], eigenvectors[0, 0]*eigenvalues[0], eigenvectors[1, 0]*eigenvalues[0], scale=5, color='r', label='Eigenvector 1')
    plt.quiver(mean[0], mean[1], eigenvectors[0, 1]*eigenvalues[1], eigenvectors[1, 1]*eigenvalues[1], scale=5, color='b', label='Eigenvector 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data Points and Eigenvectors')
    plt.legend()
    plt.axis('equal') 
    plt.grid(True)
    plt.show()

# Plot data points and eigenvectors
plot_data_and_eigenvectors(samples, mean_sample, eigenvectors, eigenvalues)

#5
# Task 5: Rotate the sample points using the eigenvectors and plot them
U_transpose = eigenvectors.T  # Transpose of the eigenvectors matrix
centered_samples = samples - mean_sample
rotated_samples = centered_samples @ U_transpose

def plot_rotated_samples(rotated_samples):
    plt.figure(figsize=(8, 8))
    plt.scatter(rotated_samples[:, 0], rotated_samples[:, 1], label='Rotated Sample Points')
    plt.xlabel('Rotated X1')
    plt.ylabel('Rotated X2')
    plt.title('Rotated Sample Points')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plot_rotated_samples(rotated_samples)
