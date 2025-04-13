# AI Training Simulator

An educational tool for learning about AI and neural network training. This program allows users to create mock datasets, train simple neural networks, and visualize how the weights change during the training process.

## Overview

This simulator is designed to help you understand:

- How neural networks learn from data
- How different datasets affect learning
- How network architecture impacts performance
- How weights evolve during training
- How decision boundaries are formed
- How words are weighted and grouped in text learning

The project includes a simple neural network implementation from scratch, various dataset generators, text processing capabilities, and comprehensive visualizations to make the learning process transparent and intuitive.

## Installation

### Prerequisites

- Python 3.6 or higher
- NumPy
- Matplotlib
- scikit-learn (for text learning features)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-training-simulator.git
   cd ai-training-simulator
   ```

2. Install the required dependencies:
   ```
   pip install numpy matplotlib scikit-learn
   ```

## Usage

### Basic Usage (Numerical Data)

Run the simulator with default settings:

```bash
python main.py
```

This will:
1. Generate a linear dataset with 100 samples
2. Create a neural network with one hidden layer (4 neurons)
3. Train the network for 1000 epochs
4. Display the results

### Text Learning

To explore how neural networks learn from text data:

```bash
python text_learning.py
```

This will:
1. Load sample text data (or generate synthetic text)
2. Process the text and create word embeddings
3. Visualize how words are weighted and grouped
4. Allow interactive exploration of word relationships

### Command Line Options (Numerical Data)

The simulator supports various command line options:

```bash
python main.py --dataset xor --samples 200 --noise 0.05 --hidden-layers 8 4 --learning-rate 0.02 --epochs 2000 --visualize
```

Options:
- `--dataset`: Type of dataset to generate (`linear`, `xor`, `circle`, `spiral`, `custom`)
- `--samples`: Number of samples in the dataset
- `--noise`: Amount of noise to add to the dataset (0.0 to 1.0)
- `--hidden-layers`: Number of neurons in each hidden layer (space-separated for multiple layers)
- `--learning-rate`: Learning rate for training
- `--epochs`: Number of training epochs
- `--visualize`: Enable visualization of the training process

### Command Line Options (Text Learning)

The text learning simulator also supports various options:

```bash
python text_learning.py --data-source sample --embedding-dim 100 --min-word-freq 3 --window-size 3 --visualize-method tsne --n-clusters 7
```

Options:
- `--data-source`: Source of text data (`sample` or `generate`)
- `--embedding-dim`: Dimensionality of word embeddings
- `--min-word-freq`: Minimum word frequency for vocabulary
- `--window-size`: Context window size for co-occurrence
- `--visualize-method`: Method for visualizing word embeddings (`pca` or `tsne`)
- `--n-clusters`: Number of word clusters to visualize

### Interactive Mode

Both simulators feature interactive modes that allow you to experiment with the models after training.

## Components

### Neural Network (`neural_network.py`)

A simple implementation of a feedforward neural network with:
- Configurable number of hidden layers and neurons
- Sigmoid activation function
- Mean squared error loss function
- Backpropagation for learning
- Weight history tracking for visualization

### Dataset Generation (`dataset.py`)

Functions for generating various types of numerical datasets:
- `linear`: Linearly separable data
- `xor`: XOR pattern (not linearly separable)
- `circle`: Points inside or outside a circle
- `spiral`: Intertwined spiral pattern
- `custom`: Multiple clusters

Each dataset can be generated with different numbers of samples and noise levels.

### Text Processing (`text_processor.py`)

Tools for processing text data and creating word embeddings:
- Building vocabulary from text
- Creating co-occurrence matrices
- Generating word embeddings using SVD
- Finding similar words
- Visualizing word relationships and clusters

### Visualization (`visualizer.py`)

Comprehensive visualization tools:
- Decision boundary visualization
- Training process animation
- Weight distribution histograms
- Weight change plots
- 3D decision boundary visualization
- Neuron activation visualization
- Word embedding visualizations

## Educational Content

### How Neural Networks Learn

1. **Initialization**: The network starts with random weights.
2. **Forward Pass**: Input data is passed through the network to produce predictions.
3. **Loss Calculation**: The difference between predictions and actual targets is calculated.
4. **Backward Pass**: The gradient of the loss with respect to each weight is calculated.
5. **Weight Update**: Weights are adjusted to reduce the loss.
6. **Iteration**: Steps 2-5 are repeated for multiple epochs.

### Understanding Word Embeddings

1. **Co-occurrence Matrix**: Counts how often words appear near each other in text.
2. **Dimensionality Reduction**: Reduces the large, sparse co-occurrence matrix to dense vectors.
3. **Word Vectors**: Each word is represented by a vector in a continuous vector space.
4. **Semantic Relationships**: Similar words have similar vectors, and relationships between words are preserved.

### Understanding the Visualizations

- **Decision Boundary**: Shows how the network separates different classes in the feature space.
- **Training Animation**: Shows how the decision boundary evolves during training.
- **Weight Distribution**: Shows the distribution of weight values in the network.
- **Weight Changes**: Shows how much the weights change during training.
- **Word Clusters**: Shows how words are grouped based on semantic similarity.
- **Word Weights**: Shows the importance of different words based on frequency and vector magnitude.

### Factors Affecting Learning

- **Dataset Complexity**: More complex patterns require more complex networks.
- **Network Architecture**: More layers and neurons can learn more complex patterns.
- **Learning Rate**: Controls how quickly the network adapts to the data.
- **Noise Level**: Higher noise makes learning more difficult.
- **Training Duration**: More epochs generally lead to better performance, up to a point.
- **Vocabulary Size**: Affects the quality of word embeddings.
- **Context Window Size**: Affects which words are considered related.

## Examples

### Numerical Data Examples

#### Linear Dataset

A simple dataset where classes can be separated by a straight line:
```bash
python main.py --dataset linear --visualize
```

#### XOR Dataset

A classic example of a problem that requires a hidden layer:
```bash
python main.py --dataset xor --hidden-layers 4 --visualize
```

#### Circle Dataset

A dataset where points are classified based on their distance from the origin:
```bash
python main.py --dataset circle --hidden-layers 8 --visualize
```

#### Spiral Dataset

A challenging dataset requiring a more complex network:
```bash
python main.py --dataset spiral --hidden-layers 16 8 --learning-rate 0.03 --epochs 3000 --visualize
```

### Text Learning Examples

#### Basic Word Embeddings

Create and visualize basic word embeddings:
```bash
python text_learning.py
```

#### Higher-Dimensional Embeddings

Create more expressive word embeddings:
```bash
python text_learning.py --embedding-dim 100 --window-size 3
```

#### Word Clustering

Explore how words cluster together:
```bash
python text_learning.py --n-clusters 7
```

## Advanced Usage

### Custom Network Architectures

Try different network architectures to see how they affect learning:
```bash
# Deep network with many small layers
python main.py --dataset spiral --hidden-layers 10 8 6 4 --visualize

# Wide network with one large layer
python main.py --dataset spiral --hidden-layers 32 --visualize
```

### Experimenting with Noise

See how different noise levels affect learning:
```bash
# No noise
python main.py --dataset xor --noise 0.0 --visualize

# High noise
python main.py --dataset xor --noise 0.3 --visualize
```

### Learning Rate Experiments

Try different learning rates to see their effect:
```bash
# Small learning rate
python main.py --dataset circle --learning-rate 0.001 --visualize

# Large learning rate
python main.py --dataset circle --learning-rate 0.1 --visualize
```

## Contributing

Contributions are welcome! Here are some ideas for improvements:

- Add more dataset types
- Implement different activation functions
- Add regularization options
- Implement batch training
- Add more visualization types
- Create a GUI interface
- Extend text learning capabilities
- Add support for pre-trained word embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details.