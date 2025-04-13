#!/usr/bin/env python3
"""
AI Training Simulator
--------------------
An educational tool for learning about AI and neural network training.
This program allows users to create mock datasets, train simple neural networks,
and visualize how the weights change during the training process.
"""

import argparse
import sys
import numpy as np
from neural_network import SimpleNeuralNetwork
from dataset import generate_dataset, add_to_dataset
from visualizer import visualize_training, plot_decision_boundary

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Training Simulator')
    parser.add_argument('--dataset', type=str, default='linear',
                        choices=['linear', 'xor', 'circle', 'spiral', 'custom'],
                        help='Type of dataset to generate')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples in the dataset')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Amount of noise to add to the dataset')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[4],
                        help='Number of neurons in each hidden layer')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the training process')
    return parser.parse_args()

def main():
    """Main function to run the AI training simulator."""
    args = parse_arguments()
    
    print("=" * 50)
    print("AI Training Simulator")
    print("=" * 50)
    
    # Generate dataset
    print(f"\nGenerating {args.dataset} dataset with {args.samples} samples and {args.noise} noise...")
    X, y = generate_dataset(args.dataset, args.samples, args.noise)
    
    # Create neural network
    input_size = X.shape[1]
    output_size = 1 if len(y.shape) == 1 else y.shape[1]
    
    print(f"\nCreating neural network with architecture: {input_size} -> {args.hidden_layers} -> {output_size}")
    nn = SimpleNeuralNetwork(input_size, args.hidden_layers, output_size, args.learning_rate)
    
    # Train the network
    print(f"\nTraining for {args.epochs} epochs...")
    history = nn.train(X, y, args.epochs, verbose=True, store_weights=args.visualize)
    
    # Evaluate the network
    accuracy = nn.evaluate(X, y)
    print(f"\nFinal accuracy: {accuracy:.2f}%")
    
    # Visualize the training process
    if args.visualize:
        print("\nVisualizing training process...")
        visualize_training(nn, history, X, y)
        plot_decision_boundary(nn, X, y)
    
    print("\nTraining complete! You can now experiment with the model.")
    
    # Interactive mode
    if args.visualize:
        print("\nEntering interactive mode. Press Ctrl+C to exit.")
        try:
            while True:
                print("\nOptions:")
                print("1. Add new data points")
                print("2. Retrain the model")
                print("3. Visualize current state")
                print("4. Exit")
                
                choice = input("\nEnter your choice (1-4): ")
                
                if choice == '1':
                    num_points = int(input("How many points to add? "))
                    new_X, new_y = add_to_dataset(args.dataset, num_points, args.noise)
                    X = np.vstack([X, new_X])
                    y = np.concatenate([y, new_y])
                    print(f"Added {num_points} new points. Dataset now has {len(y)} samples.")
                
                elif choice == '2':
                    epochs = int(input("How many epochs to train? "))
                    history = nn.train(X, y, epochs, verbose=True, store_weights=True)
                    accuracy = nn.evaluate(X, y)
                    print(f"New accuracy: {accuracy:.2f}%")
                
                elif choice == '3':
                    visualize_training(nn, history, X, y)
                    plot_decision_boundary(nn, X, y)
                
                elif choice == '4':
                    break
                
                else:
                    print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
    
    print("\nThank you for using the AI Training Simulator!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)