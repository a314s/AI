#!/usr/bin/env python3
"""
Text Learning Simulator
---------------------
A script to demonstrate how neural networks learn from text data,
showing how words get weighted and grouped during training.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from text_processor import TextProcessor, load_sample_texts, generate_text_dataset
from neural_network import SimpleNeuralNetwork

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Text Learning Simulator')
    parser.add_argument('--data-source', type=str, default='sample',
                        choices=['sample', 'generate'],
                        help='Source of text data (sample or generate)')
    parser.add_argument('--embedding-dim', type=int, default=50,
                        help='Dimensionality of word embeddings')
    parser.add_argument('--min-word-freq', type=int, default=2,
                        help='Minimum word frequency for vocabulary')
    parser.add_argument('--window-size', type=int, default=2,
                        help='Context window size for co-occurrence')
    parser.add_argument('--visualize-method', type=str, default='tsne',
                        choices=['pca', 'tsne'],
                        help='Method for visualizing word embeddings')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of word clusters to visualize')
    return parser.parse_args()

def main():
    """Main function to run the text learning simulator."""
    args = parse_arguments()
    
    print("=" * 50)
    print("Text Learning Simulator")
    print("=" * 50)
    
    # Load or generate text data
    if args.data_source == 'sample':
        print("\nLoading sample texts...")
        texts = load_sample_texts()
    else:
        print("\nGenerating synthetic text dataset...")
        texts, _ = generate_text_dataset(n_samples=100, vocab_size=500, seq_length=20, n_classes=2)
    
    print(f"Loaded {len(texts)} texts")
    print(f"Sample text: {texts[0][:100]}...")
    
    # Initialize text processor
    print("\nInitializing text processor...")
    text_processor = TextProcessor(
        min_word_freq=args.min_word_freq,
        max_vocab_size=5000,
        window_size=args.window_size
    )
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    text_processor.build_vocabulary(texts)
    
    # Build co-occurrence matrix
    print("\nBuilding co-occurrence matrix...")
    text_processor.build_co_occurrence_matrix(texts)
    
    # Create word embeddings
    print(f"\nCreating word embeddings with dimension {args.embedding_dim}...")
    word_vectors = text_processor.create_word_vectors(embedding_dim=args.embedding_dim)
    
    # Display some statistics
    print(f"\nVocabulary size: {text_processor.vocab_size}")
    print(f"Word vector shape: {word_vectors.shape}")
    
    # Visualize word embeddings
    print(f"\nVisualizing word embeddings using {args.visualize_method}...")
    text_processor.visualize_word_vectors(method=args.visualize_method)
    
    # Visualize word clusters
    print(f"\nVisualizing word clusters (k={args.n_clusters})...")
    text_processor.visualize_word_clusters(n_clusters=args.n_clusters)
    
    # Visualize word weights
    print("\nVisualizing word weights...")
    text_processor.visualize_word_weights()
    
    # Interactive mode
    print("\nEntering interactive mode. Press Ctrl+C to exit.")
    try:
        while True:
            print("\nOptions:")
            print("1. Find similar words")
            print("2. Visualize specific words")
            print("3. Explain word embeddings")
            print("4. Train a text classifier")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                query_word = input("Enter a word: ")
                similar_words = text_processor.get_similar_words(query_word, top_n=10)
                
                if similar_words:
                    print(f"\nWords similar to '{query_word}':")
                    for word, similarity in similar_words:
                        print(f"  {word}: {similarity:.4f}")
                else:
                    print(f"Word '{query_word}' not found in vocabulary.")
            
            elif choice == '2':
                words_input = input("Enter words to visualize (comma-separated): ")
                words = [word.strip() for word in words_input.split(',')]
                
                if words:
                    text_processor.visualize_word_vectors(words=words, method=args.visualize_method)
                else:
                    print("No words provided.")
            
            elif choice == '3':
                explanation = text_processor.explain_word_embeddings()
                print(explanation)
            
            elif choice == '4':
                train_text_classifier(text_processor, texts)
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
    
    print("\nThank you for using the Text Learning Simulator!")

def train_text_classifier(text_processor, texts):
    """
    Train a simple neural network to classify text.
    
    Args:
        text_processor: Initialized TextProcessor with word embeddings
        texts: List of text strings
    """
    # Generate synthetic labels for demonstration
    print("\nGenerating synthetic classification task...")
    labels = np.random.randint(0, 2, size=len(texts))
    
    # Convert texts to fixed-length sequences of word indices
    print("Converting texts to sequences...")
    max_seq_length = 20  # Truncate/pad to this length
    
    # Create feature vectors by averaging word embeddings for each text
    print("Creating feature vectors from word embeddings...")
    X = np.zeros((len(texts), text_processor.word_vectors.shape[1]))
    
    for i, text in enumerate(texts):
        tokens = text_processor.preprocess_text(text)
        token_indices = [text_processor.word_to_idx[token] for token in tokens 
                         if token in text_processor.word_to_idx]
        
        if token_indices:
            # Average the word vectors for this text
            text_vector = np.mean(text_processor.word_vectors[token_indices], axis=0)
            X[i] = text_vector
    
    # Split into train/test
    train_size = int(0.8 * len(texts))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    
    # Create and train a simple neural network
    print("\nTraining neural network classifier...")
    input_size = X_train.shape[1]
    hidden_layers = [10]
    output_size = 1  # Binary classification
    
    nn = SimpleNeuralNetwork(input_size, hidden_layers, output_size, learning_rate=0.01)
    history = nn.train(X_train, y_train, epochs=200, verbose=True)
    
    # Evaluate
    accuracy = nn.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    
    # Get the weights from the first layer
    weights = nn.weights[0]  # Shape: (input_size, hidden_size)
    
    # Calculate the average absolute weight for each input feature
    feature_importance = np.mean(np.abs(weights), axis=1)
    
    # Get the top 20 most important features
    top_indices = np.argsort(feature_importance)[-20:]
    
    # Find words that contribute most to these features
    print("\nTop words by feature importance:")
    
    # For each input feature (embedding dimension)
    for feature_idx in reversed(top_indices):
        # Find words with highest value in this dimension
        word_values = [(word, text_processor.word_vectors[text_processor.word_to_idx[word]][feature_idx]) 
                      for word in text_processor.word_to_idx.keys()]
        
        # Sort by absolute value
        word_values.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Print top 5 words for this feature
        top_words = word_values[:5]
        print(f"Feature {feature_idx}: " + ", ".join([f"{word} ({value:.4f})" for word, value in top_words]))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")