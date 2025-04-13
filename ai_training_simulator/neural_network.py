"""
Neural Network Implementation
----------------------------
A simple implementation of a feedforward neural network with backpropagation.
This module is designed for educational purposes to demonstrate how neural networks
learn from data and how weights are updated during training.
"""

import numpy as np
import time

class SimpleNeuralNetwork:
    """A simple feedforward neural network implementation for educational purposes."""
    
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_layers (list): List of integers representing the number of neurons in each hidden layer
            output_size (int): Number of output neurons
            learning_rate (float): Learning rate for weight updates
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize network architecture
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Initialize weights with small random values
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            self.weights.append(w)
            
            # Initialize biases with zeros
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.biases.append(b)
        
        # For visualization and educational purposes
        self.weight_history = []
        self.accuracy_history = []
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, input_size)
            
        Returns:
            list: List of activations for each layer
        """
        activations = [X]
        
        # Forward pass through each layer
        for i in range(len(self.weights)):
            # Calculate the weighted sum
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Apply activation function
            a = self.sigmoid(z)
            
            # Store the activation
            activations.append(a)
        
        return activations
    
    def backward(self, X, y, activations):
        """
        Backward pass to update weights using backpropagation.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, input_size)
            y (numpy.ndarray): Target data of shape (n_samples, output_size)
            activations (list): List of activations from forward pass
            
        Returns:
            float: Loss value
        """
        m = X.shape[0]  # Number of samples
        
        # Convert y to appropriate shape if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Calculate output layer error
        output_error = activations[-1] - y
        loss = np.mean(np.square(output_error))
        
        # Backpropagate the error
        delta = output_error
        
        # Update weights and biases for each layer (from output to input)
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradient
            gradient = np.dot(activations[i].T, delta) / m
            
            # Update weights
            self.weights[i] -= self.learning_rate * gradient
            
            # Update biases
            self.biases[i] -= self.learning_rate * np.mean(delta, axis=0, keepdims=True)
            
            # Calculate error for the previous layer (if not the input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        return loss
    
    def train(self, X, y, epochs, verbose=False, store_weights=False):
        """
        Train the neural network.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, input_size)
            y (numpy.ndarray): Target data of shape (n_samples, output_size)
            epochs (int): Number of training epochs
            verbose (bool): Whether to print progress
            store_weights (bool): Whether to store weight history for visualization
            
        Returns:
            dict: Training history
        """
        history = {
            'loss': [],
            'accuracy': [],
            'weights': [] if store_weights else None
        }
        
        # Reset weight history
        self.weight_history = []
        self.accuracy_history = []
        self.loss_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            
            # Backward pass
            loss = self.backward(X, y, activations)
            
            # Calculate accuracy
            predictions = self.predict(X)
            if len(y.shape) == 1:
                accuracy = np.mean((predictions.flatten() > 0.5) == y) * 100
            else:
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) * 100
            
            # Store history
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            # Store weights for visualization if requested
            if store_weights and (epoch % max(1, epochs // 100) == 0):
                self.weight_history.append([w.copy() for w in self.weights])
                self.accuracy_history.append(accuracy)
                self.loss_history.append(loss)
            
            # Print progress
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.2f}%")
                
                # Add a small delay to make the progress visible
                time.sleep(0.01)
        
        # Store final weights
        if store_weights:
            self.weight_history.append([w.copy() for w in self.weights])
            self.accuracy_history.append(accuracy)
            self.loss_history.append(loss)
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained network.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, input_size)
            
        Returns:
            numpy.ndarray: Predictions
        """
        # Forward pass
        activations = self.forward(X)
        
        # Return the output layer activations
        return activations[-1]
    
    def evaluate(self, X, y):
        """
        Evaluate the network on test data.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, input_size)
            y (numpy.ndarray): Target data of shape (n_samples, output_size)
            
        Returns:
            float: Accuracy percentage
        """
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate accuracy
        if len(y.shape) == 1:
            accuracy = np.mean((predictions.flatten() > 0.5) == y) * 100
        else:
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) * 100
        
        return accuracy
    
    def get_weights(self):
        """Get the current weights of the network."""
        return self.weights
    
    def get_weight_history(self):
        """Get the weight history during training."""
        return self.weight_history
    
    def get_accuracy_history(self):
        """Get the accuracy history during training."""
        return self.accuracy_history
    
    def get_loss_history(self):
        """Get the loss history during training."""
        return self.loss_history
    
    def explain_weights(self):
        """
        Provide an explanation of the current weights.
        This is an educational function to help understand what the weights represent.
        
        Returns:
            str: Explanation of weights
        """
        explanation = []
        explanation.append("Neural Network Weight Explanation:")
        explanation.append("=" * 40)
        
        for i, w in enumerate(self.weights):
            if i == 0:
                layer_type = "Input-to-Hidden"
            elif i == len(self.weights) - 1:
                layer_type = "Hidden-to-Output"
            else:
                layer_type = f"Hidden-to-Hidden ({i})"
            
            explanation.append(f"\nLayer {i+1} ({layer_type}):")
            explanation.append(f"  Shape: {w.shape}")
            explanation.append(f"  Mean: {w.mean():.4f}")
            explanation.append(f"  Std Dev: {w.std():.4f}")
            explanation.append(f"  Min: {w.min():.4f}")
            explanation.append(f"  Max: {w.max():.4f}")
            
            # For small networks, show some example weights
            if w.size < 100:
                explanation.append("  Sample weights:")
                for j in range(min(5, w.shape[0])):
                    row_str = "    "
                    for k in range(min(5, w.shape[1])):
                        row_str += f"{w[j, k]:.4f} "
                    explanation.append(row_str)
        
        return "\n".join(explanation)