"""
Visualization Module
------------------
Functions for visualizing neural network training, decision boundaries,
and weight changes during the learning process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time

def visualize_training(nn, history, X, y):
    """
    Visualize the training process of a neural network.
    
    Args:
        nn: The trained neural network
        history: Training history dictionary
        X: Input data
        y: Target data
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # Plot 1: Loss curve
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(history['loss'], 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curve
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(history['accuracy'], 'g-')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weight distribution
    ax3 = plt.subplot(gs[0, 2])
    visualize_weight_distribution(nn, ax3)
    
    # Plot 4: Decision boundary
    ax4 = plt.subplot(gs[1, :2])
    plot_decision_boundary_on_axis(nn, X, y, ax4)
    
    # Plot 5: Weight changes over time
    ax5 = plt.subplot(gs[1, 2])
    visualize_weight_changes(nn, ax5)
    
    plt.tight_layout()
    plt.show()
    
    # If we have weight history, create an animation
    if nn.get_weight_history():
        animate_training(nn, X, y)

def visualize_weight_distribution(nn, ax=None):
    """
    Visualize the distribution of weights in the neural network.
    
    Args:
        nn: The trained neural network
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    weights = nn.get_weights()
    all_weights = np.concatenate([w.flatten() for w in weights])
    
    ax.hist(all_weights, bins=50, alpha=0.7, color='purple')
    ax.set_title('Weight Distribution')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.95, f"Mean: {all_weights.mean():.4f}\nStd: {all_weights.std():.4f}",
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    return ax

def visualize_weight_changes(nn, ax=None):
    """
    Visualize how weights changed during training.
    
    Args:
        nn: The trained neural network
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    weight_history = nn.get_weight_history()
    
    if not weight_history:
        ax.text(0.5, 0.5, "No weight history available.\nEnable store_weights=True during training.",
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Calculate the average absolute weight change for each layer
    changes = []
    for i in range(len(weight_history) - 1):
        layer_changes = []
        for layer_idx in range(len(weight_history[0])):
            change = np.mean(np.abs(weight_history[i+1][layer_idx] - weight_history[i][layer_idx]))
            layer_changes.append(change)
        changes.append(layer_changes)
    
    changes = np.array(changes)
    
    # Plot changes for each layer
    for layer_idx in range(changes.shape[1]):
        ax.plot(changes[:, layer_idx], label=f'Layer {layer_idx+1}')
    
    ax.set_title('Weight Changes During Training')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Avg. Absolute Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_decision_boundary(nn, X, y):
    """
    Plot the decision boundary of a trained neural network.
    
    Args:
        nn: The trained neural network
        X: Input data
        y: Target data
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_boundary_on_axis(nn, X, y, ax)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary_on_axis(nn, X, y, ax):
    """
    Plot the decision boundary on a given axis.
    
    Args:
        nn: The trained neural network
        X: Input data
        y: Target data
        ax: Matplotlib axis to plot on
    """
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create a meshgrid
    h = 0.01  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for each point in the meshgrid
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Reshape the predictions
    if Z.shape[1] > 1:  # Multi-class
        Z = np.argmax(Z, axis=1)
    else:  # Binary
        Z = (Z > 0.5).astype(int).flatten()
    
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Add a legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    # Add accuracy information
    accuracy = nn.evaluate(X, y)
    ax.text(0.02, 0.02, f"Accuracy: {accuracy:.2f}%",
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    return ax

def animate_training(nn, X, y):
    """
    Create an animation showing how the decision boundary evolved during training.
    
    Args:
        nn: The trained neural network
        X: Input data
        y: Target data
    """
    weight_history = nn.get_weight_history()
    accuracy_history = nn.get_accuracy_history()
    loss_history = nn.get_loss_history()
    
    if not weight_history:
        print("No weight history available. Enable store_weights=True during training.")
        return
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Decision boundary plot
    ax1 = plt.subplot(gs[:, 0])
    ax1.set_title('Decision Boundary Evolution')
    
    # Loss plot
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax3 = plt.subplot(gs[1, 1])
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create a meshgrid
    h = 0.02  # Step size (coarser for animation speed)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Plot the data points (these don't change)
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    # Add a legend
    legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
    ax1.add_artist(legend1)
    
    # Set axis limits
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # Initialize contour plot
    contour = [ax1.contourf(xx, yy, np.zeros_like(xx), alpha=0.3, cmap=plt.cm.coolwarm)]
    
    # Initialize loss and accuracy plots
    loss_line, = ax2.plot([], [], 'b-')
    acc_line, = ax3.plot([], [], 'g-')
    
    # Initialize text for step and accuracy
    step_text = ax1.text(0.02, 0.02, "", transform=ax1.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.7))
    
    # Set y-axis limits for loss and accuracy
    if loss_history:
        ax2.set_ylim(0, max(loss_history) * 1.1)
        ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    
    def init():
        """Initialize the animation."""
        contour[0] = ax1.contourf(xx, yy, np.zeros_like(xx), alpha=0.3, cmap=plt.cm.coolwarm)
        loss_line.set_data([], [])
        acc_line.set_data([], [])
        step_text.set_text("")
        return contour[0], loss_line, acc_line, step_text
    
    def animate(i):
        """Update the animation for frame i."""
        # Clear previous contour
        for coll in contour[0].collections:
            coll.remove()
        
        # Create a temporary neural network with the historical weights
        nn._weights = weight_history[i]
        
        # Predict for each point in the meshgrid
        Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Reshape the predictions
        if Z.shape[1] > 1:  # Multi-class
            Z = np.argmax(Z, axis=1)
        else:  # Binary
            Z = (Z > 0.5).astype(int).flatten()
        
        Z = Z.reshape(xx.shape)
        
        # Update the contour plot
        contour[0] = ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Update the loss and accuracy plots
        x_data = np.arange(i+1)
        loss_line.set_data(x_data, loss_history[:i+1])
        acc_line.set_data(x_data, accuracy_history[:i+1])
        
        # Update the step and accuracy text
        step_text.set_text(f"Step: {i}/{len(weight_history)-1}\nAccuracy: {accuracy_history[i]:.2f}%")
        
        return contour[0], loss_line, acc_line, step_text
    
    # Create the animation
    frames = len(weight_history)
    interval = 200  # milliseconds between frames
    
    anim = FuncAnimation(fig, animate, frames=frames, init_func=init,
                         interval=interval, blit=False)
    
    plt.show()

def visualize_3d_decision_boundary(nn, X, y):
    """
    Create a 3D visualization of the decision boundary.
    
    Args:
        nn: The trained neural network
        X: Input data
        y: Target data
    """
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create a meshgrid
    h = 0.01  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for each point in the meshgrid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(mesh_points)
    
    if Z.shape[1] > 1:  # Multi-class
        Z = np.argmax(Z, axis=1)
    else:  # Binary
        Z = Z.flatten()
    
    Z = Z.reshape(xx.shape)
    
    # Plot the surface
    surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)
    
    # Plot the data points
    for i in np.unique(y):
        idx = (y == i)
        ax.scatter(X[idx, 0], X[idx, 1], y[idx], c=['b', 'r'][int(i)], label=f'Class {i}', s=20, alpha=0.6)
    
    ax.set_title('3D Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Prediction')
    ax.legend()
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

def visualize_neuron_activations(nn, X):
    """
    Visualize the activations of neurons in each layer for a given input.
    
    Args:
        nn: The neural network
        X: Input data (single sample or batch)
    """
    # If X is a batch, use just the first sample for visualization
    if len(X.shape) > 1 and X.shape[0] > 1:
        X_sample = X[0:1]
    else:
        X_sample = X
    
    # Get activations for each layer
    activations = nn.forward(X_sample)
    
    # Create a figure
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
    
    # If there's only one layer, wrap the axis in a list
    if n_layers == 1:
        axes = [axes]
    
    # Plot activations for each layer
    for i, activation in enumerate(activations):
        ax = axes[i]
        
        # Reshape activation for visualization
        act_flat = activation.flatten()
        
        # Plot as a bar chart
        ax.bar(range(len(act_flat)), act_flat, alpha=0.7)
        
        # Set title and labels
        if i == 0:
            ax.set_title('Input Layer')
        elif i == n_layers - 1:
            ax.set_title('Output Layer')
        else:
            ax.set_title(f'Hidden Layer {i}')
        
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Activation')
        
        # Add statistics
        ax.text(0.02, 0.95, f"Mean: {act_flat.mean():.4f}\nMax: {act_flat.max():.4f}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def explain_visualization(visualization_type):
    """
    Provide an explanation of a visualization type.
    
    Args:
        visualization_type (str): Type of visualization to explain
        
    Returns:
        str: Explanation of the visualization
    """
    explanations = {
        'decision_boundary': """
        Decision Boundary Visualization:
        ------------------------------
        This visualization shows how the neural network separates different classes in the feature space.
        
        The colored regions represent areas where the network predicts different classes.
        The boundary between colors is the decision boundary - where the network's prediction changes from one class to another.
        
        Points are colored according to their true class, so you can see which points are correctly classified
        (when the point color matches the background color) and which are misclassified.
        
        For a well-trained network, most points should be in regions matching their color.
        """,
        
        'training_animation': """
        Training Animation:
        -----------------
        This animation shows how the decision boundary evolves during training.
        
        You can observe:
        1. How the boundary starts simple and becomes more complex
        2. How the network gradually learns to separate the classes
        3. How the loss decreases and accuracy increases over time
        
        This visualization helps understand the learning process and how the network
        gradually adapts its weights to fit the data.
        """,
        
        'weight_distribution': """
        Weight Distribution:
        ------------------
        This histogram shows the distribution of weight values in the neural network.
        
        In a well-initialized and well-trained network:
        - Weights should be centered around zero
        - The distribution should be roughly symmetric
        - Extreme values should be rare
        
        If weights become too large (weight explosion) or too small (vanishing weights),
        it can indicate training problems.
        """,
        
        'weight_changes': """
        Weight Changes:
        -------------
        This plot shows how much the weights change during training.
        
        Typically:
        - Changes are larger at the beginning of training
        - Changes decrease as training progresses
        - Different layers may have different patterns of change
        
        If changes remain large throughout training, it might indicate that the
        learning rate is too high or the network is unstable.
        """,
        
        'neuron_activations': """
        Neuron Activations:
        -----------------
        This visualization shows the activation values of neurons in each layer for a specific input.
        
        You can observe:
        1. How the input is transformed through the network
        2. Which neurons are active (high activation) or inactive (low activation)
        3. How information is propagated and transformed
        
        In a well-functioning network, activations should be diverse - some neurons active,
        some inactive - rather than all neurons having similar activation values.
        """
    }
    
    return explanations.get(visualization_type, "No explanation available for this visualization type.")