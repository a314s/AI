"""
Dataset Generation Module
------------------------
Functions for generating and manipulating mock datasets for neural network training.
This module provides various dataset types with adjustable parameters to demonstrate
how different data distributions affect neural network learning.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(dataset_type, n_samples=100, noise=0.1):
    """
    Generate a mock dataset for neural network training.
    
    Args:
        dataset_type (str): Type of dataset to generate ('linear', 'xor', 'circle', 'spiral', 'custom')
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    if dataset_type == 'linear':
        return generate_linear_dataset(n_samples, noise)
    elif dataset_type == 'xor':
        return generate_xor_dataset(n_samples, noise)
    elif dataset_type == 'circle':
        return generate_circle_dataset(n_samples, noise)
    elif dataset_type == 'spiral':
        return generate_spiral_dataset(n_samples, noise)
    elif dataset_type == 'custom':
        return generate_custom_dataset(n_samples, noise)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def generate_linear_dataset(n_samples=100, noise=0.1):
    """
    Generate a linearly separable dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Generate random points in 2D space
    X = np.random.rand(n_samples, 2) * 2 - 1  # Values between -1 and 1
    
    # Create a linear boundary: y = m*x + b
    m, b = 0.5, 0.1
    
    # Determine the class based on the boundary
    y_clean = (X[:, 1] > m * X[:, 0] + b).astype(int)
    
    # Add noise by flipping some labels
    if noise > 0:
        flip_indices = np.random.choice(
            n_samples, 
            size=int(n_samples * noise), 
            replace=False
        )
        y = y_clean.copy()
        y[flip_indices] = 1 - y[flip_indices]  # Flip the labels
    else:
        y = y_clean
    
    print(f"Generated linear dataset with {n_samples} samples")
    print(f"  - Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")
    print(f"  - Noise level: {noise} ({len(flip_indices) if noise > 0 else 0} flipped labels)")
    
    return X, y

def generate_xor_dataset(n_samples=100, noise=0.1):
    """
    Generate an XOR dataset (not linearly separable).
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Generate random points in 2D space
    X = np.random.rand(n_samples, 2) * 2 - 1  # Values between -1 and 1
    
    # XOR function: (x > 0) XOR (y > 0)
    y_clean = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    # Add noise by flipping some labels
    if noise > 0:
        flip_indices = np.random.choice(
            n_samples, 
            size=int(n_samples * noise), 
            replace=False
        )
        y = y_clean.copy()
        y[flip_indices] = 1 - y[flip_indices]  # Flip the labels
    else:
        y = y_clean
    
    print(f"Generated XOR dataset with {n_samples} samples")
    print(f"  - Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")
    print(f"  - Noise level: {noise} ({len(flip_indices) if noise > 0 else 0} flipped labels)")
    
    return X, y

def generate_circle_dataset(n_samples=100, noise=0.1):
    """
    Generate a circular dataset (points inside or outside a circle).
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Generate random points in 2D space
    X = np.random.rand(n_samples, 2) * 2 - 1  # Values between -1 and 1
    
    # Calculate distance from origin
    distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    # Points inside the circle (radius 0.5) are class 1, outside are class 0
    y_clean = (distances < 0.5).astype(int)
    
    # Add noise by flipping some labels
    if noise > 0:
        flip_indices = np.random.choice(
            n_samples, 
            size=int(n_samples * noise), 
            replace=False
        )
        y = y_clean.copy()
        y[flip_indices] = 1 - y[flip_indices]  # Flip the labels
    else:
        y = y_clean
    
    print(f"Generated circle dataset with {n_samples} samples")
    print(f"  - Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")
    print(f"  - Noise level: {noise} ({len(flip_indices) if noise > 0 else 0} flipped labels)")
    
    return X, y

def generate_spiral_dataset(n_samples=100, noise=0.1):
    """
    Generate a spiral dataset with two intertwined classes.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Ensure even number of samples
    n_samples = (n_samples // 2) * 2
    
    # Generate spiral data
    n = n_samples // 2  # samples per class
    
    # Polar coordinates
    r = np.linspace(0, 1, n)  # radius
    t0 = np.linspace(0, 4*np.pi, n) + noise * np.random.randn(n)  # theta for class 0
    t1 = np.linspace(0, 4*np.pi, n) + np.pi + noise * np.random.randn(n)  # theta for class 1
    
    # Convert to Cartesian coordinates
    X0 = np.column_stack([r*np.cos(t0), r*np.sin(t0)])
    X1 = np.column_stack([r*np.cos(t1), r*np.sin(t1)])
    
    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    print(f"Generated spiral dataset with {n_samples} samples")
    print(f"  - Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")
    print(f"  - Noise level: {noise}")
    
    return X, y

def generate_custom_dataset(n_samples=100, noise=0.1):
    """
    Generate a custom dataset with multiple clusters.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Ensure divisible by 4 for balanced classes
    n_samples = (n_samples // 4) * 4
    n_per_cluster = n_samples // 4
    
    # Generate 4 clusters (2 for each class)
    centers = [
        [-0.5, -0.5],  # Class 0
        [0.5, 0.5],    # Class 0
        [-0.5, 0.5],   # Class 1
        [0.5, -0.5]    # Class 1
    ]
    
    X_list = []
    y_list = []
    
    for i, center in enumerate(centers):
        # Generate points around the center
        cluster_x = np.random.randn(n_per_cluster, 2) * 0.2 + center
        cluster_y = np.ones(n_per_cluster) * (i // 2)  # 0 for first two clusters, 1 for last two
        
        X_list.append(cluster_x)
        y_list.append(cluster_y)
    
    # Combine data
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    print(f"Generated custom dataset with {n_samples} samples")
    print(f"  - Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")
    print(f"  - Noise level: {noise}")
    
    return X, y

def add_to_dataset(dataset_type, n_samples=10, noise=0.1):
    """
    Generate additional data points for an existing dataset.
    
    Args:
        dataset_type (str): Type of dataset to generate points for
        n_samples (int): Number of new samples to generate
        noise (float): Amount of noise to add to the new samples
        
    Returns:
        tuple: (X_new, y_new) where X_new contains the new feature vectors and y_new the new targets
    """
    # Generate new data points using the same generation function
    X_new, y_new = generate_dataset(dataset_type, n_samples, noise)
    
    print(f"Generated {n_samples} new data points for {dataset_type} dataset")
    
    return X_new, y_new

def visualize_dataset(X, y, title="Dataset Visualization"):
    """
    Visualize a 2D dataset with class labels.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, 2)
        y (numpy.ndarray): Target vector of shape (n_samples,)
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot points for each class
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a note about the dataset size
    plt.text(0.02, 0.02, f"Total samples: {len(y)}\nClass 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}",
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def explain_dataset(dataset_type):
    """
    Provide an explanation of a dataset type.
    
    Args:
        dataset_type (str): Type of dataset to explain
        
    Returns:
        str: Explanation of the dataset
    """
    explanations = {
        'linear': """
        Linear Dataset:
        --------------
        This dataset contains points that can be separated by a straight line (linear boundary).
        It's the simplest type of classification problem and can be solved by:
        - A single-layer perceptron (no hidden layers)
        - Linear models like logistic regression
        
        Learning this dataset demonstrates the basic concept of finding a decision boundary.
        """,
        
        'xor': """
        XOR Dataset:
        -----------
        The XOR (exclusive OR) dataset is not linearly separable, meaning no straight line
        can separate the two classes. Points are classified as:
        - Class 1 if both coordinates have the same sign (both positive or both negative)
        - Class 0 if coordinates have different signs
        
        This dataset demonstrates why we need hidden layers in neural networks, as a
        single-layer perceptron cannot learn this pattern.
        """,
        
        'circle': """
        Circle Dataset:
        -------------
        This dataset classifies points based on their distance from the origin:
        - Points inside a circle are one class
        - Points outside the circle are another class
        
        This is another example of a non-linearly separable dataset that requires
        a neural network with at least one hidden layer to learn properly.
        """,
        
        'spiral': """
        Spiral Dataset:
        -------------
        This dataset consists of two intertwined spiral patterns, each representing
        a different class. It's a challenging dataset that requires a neural network
        with multiple hidden layers to learn effectively.
        
        The spiral pattern demonstrates how neural networks can learn complex,
        highly non-linear decision boundaries.
        """,
        
        'custom': """
        Custom Dataset:
        -------------
        This dataset contains multiple clusters of points, with each class having
        points in different regions of the feature space. It demonstrates how
        neural networks can learn disjoint decision boundaries.
        
        This type of dataset shows that a single class can occupy multiple,
        separate regions in the feature space.
        """
    }
    
    return explanations.get(dataset_type, "No explanation available for this dataset type.")