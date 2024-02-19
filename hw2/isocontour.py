import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to plot isocontours
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_isocontours(mu, sigma, title):
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    
    rv = multivariate_normal(mean=mu, cov=sigma)
    contour = plt.contour(x, y, rv.pdf(pos), cmap='viridis', levels=10)
    
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Add labels with isovalues
    plt.clabel(contour, inline=True, fontsize=8)

    # Add a colorbar
    cbar = plt.colorbar(contour, label='Probability Density')
    
    plt.grid(True)
    plt.show()

# Example usage
mu = np.array([1, 1])
sigma = np.array([[1, 0], [0, 2]])
plot_isocontours(mu, sigma, 'Example: f(µ, Σ)')

# Part 1
mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0], [0, 2]])
plot_isocontours(mu1, sigma1, 'Part 1: f(µ, Σ)')

# Part 2
mu2 = np.array([-1, 2])
sigma2 = np.array([[2, 1], [1, 4]])
plot_isocontours(mu2, sigma2, 'Part 2: f(µ, Σ)')

# Part 3
mu3_1 = np.array([0, 2])
mu3_2 = np.array([2, 0])
sigma3_1 = sigma3_2 = np.array([[2, 1], [1, 1]])
diff_pdf = lambda x: multivariate_normal(mu3_1, sigma3_1).pdf(x) - multivariate_normal(mu3_2, sigma3_2).pdf(x)
plot_isocontours(mu3_1, sigma3_1, 'Part 3: f(µ1, Σ1) - f(µ2, Σ2)')

# Part 4
mu4_1 = np.array([0, 2])
mu4_2 = np.array([2, 0])
sigma4_1 = np.array([[2, 1], [1, 1]])
sigma4_2 = np.array([[2, 1], [1, 4]])
diff_pdf2 = lambda x: multivariate_normal(mu4_1, sigma4_1).pdf(x) - multivariate_normal(mu4_2, sigma4_2).pdf(x)
plot_isocontours(mu4_1, sigma4_1, 'Part 4: f(µ1, Σ1) - f(µ2, Σ2)')

# Part 5
mu5_1 = np.array([1, 1])
mu5_2 = np.array([-1, -1])
sigma5_1 = np.array([[2, 0], [0, 1]])
sigma5_2 = np.array([[2, 1], [1, 2]])
diff_pdf3 = lambda x: multivariate_normal(mu5_1, sigma5_1).pdf(x) - multivariate_normal(mu5_2, sigma5_2).pdf(x)
plot_isocontours(mu5_1, sigma5_1, 'Part 5: f(µ1, Σ1) - f(µ2, Σ2)')
