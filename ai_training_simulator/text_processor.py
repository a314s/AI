"""
Text Processing Module
--------------------
Functions for processing text data, creating word embeddings, and visualizing
word relationships for neural network training.
"""

import numpy as np
import re
import string
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class TextProcessor:
    """A class for processing text data and creating word embeddings."""
    
    def __init__(self, min_word_freq=2, max_vocab_size=5000, window_size=2):
        """
        Initialize the text processor.
        
        Args:
            min_word_freq (int): Minimum frequency for a word to be included in vocabulary
            max_vocab_size (int): Maximum vocabulary size
            window_size (int): Context window size for co-occurrence matrix
        """
        self.min_word_freq = min_word_freq
        self.max_vocab_size = max_vocab_size
        self.window_size = window_size
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freqs = None
        self.vocab_size = 0
        self.co_occurrence_matrix = None
        self.word_vectors = None
    
    def preprocess_text(self, text):
        """
        Preprocess text by converting to lowercase, removing punctuation and numbers.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(f'[{string.punctuation}0-9]', ' ', text)
        
        # Split into tokens and remove empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Word to index mapping
        """
        # Preprocess all texts and count word frequencies
        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Count word frequencies
        word_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= self.min_word_freq}
        
        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if len(sorted_words) > self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size]
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.word_freqs = {word: count for word, count in sorted_words}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Top 10 most frequent words: {sorted_words[:10]}")
        
        return self.word_to_idx
    
    def build_co_occurrence_matrix(self, texts):
        """
        Build co-occurrence matrix from texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Co-occurrence matrix
        """
        # Initialize co-occurrence matrix
        co_occurrence = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        
        # Process each text
        for text in texts:
            tokens = self.preprocess_text(text)
            
            # Convert tokens to indices, skipping unknown words
            token_indices = [self.word_to_idx[token] for token in tokens 
                            if token in self.word_to_idx]
            
            # Build co-occurrence matrix
            for i, center_idx in enumerate(token_indices):
                # Context window
                start = max(0, i - self.window_size)
                end = min(len(token_indices), i + self.window_size + 1)
                
                # Update co-occurrence counts
                for j in range(start, end):
                    if i != j:  # Skip the word itself
                        context_idx = token_indices[j]
                        co_occurrence[center_idx, context_idx] += 1.0
        
        # Store the co-occurrence matrix
        self.co_occurrence_matrix = co_occurrence
        
        print(f"Built co-occurrence matrix of shape {co_occurrence.shape}")
        print(f"Total co-occurrences: {np.sum(co_occurrence)}")
        
        return co_occurrence
    
    def create_word_vectors(self, embedding_dim=50):
        """
        Create word vectors using SVD on the co-occurrence matrix.
        
        Args:
            embedding_dim (int): Dimensionality of word embeddings
            
        Returns:
            numpy.ndarray: Word vectors
        """
        if self.co_occurrence_matrix is None:
            raise ValueError("Co-occurrence matrix not built. Call build_co_occurrence_matrix first.")
        
        # Apply log to co-occurrence counts (helps with the distribution)
        log_co_occurrence = np.log(self.co_occurrence_matrix + 1)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(log_co_occurrence, full_matrices=False)
        
        # Take the first embedding_dim components
        self.word_vectors = U[:, :embedding_dim]
        
        print(f"Created word vectors with dimension {embedding_dim}")
        
        return self.word_vectors
    
    def get_similar_words(self, word, top_n=10):
        """
        Find most similar words to a given word based on vector similarity.
        
        Args:
            word (str): Query word
            top_n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        if self.word_vectors is None:
            raise ValueError("Word vectors not created. Call create_word_vectors first.")
        
        if word not in self.word_to_idx:
            return []
        
        # Get the word vector
        word_idx = self.word_to_idx[word]
        word_vec = self.word_vectors[word_idx]
        
        # Compute cosine similarity with all other words
        norm = np.linalg.norm(self.word_vectors, axis=1) * np.linalg.norm(word_vec)
        similarities = np.dot(self.word_vectors, word_vec) / (norm + 1e-8)
        
        # Get top similar words (excluding the query word)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        similar_words = [(self.idx_to_word[idx], similarities[idx]) for idx in similar_indices]
        
        return similar_words
    
    def visualize_word_vectors(self, words=None, method='tsne', perplexity=30):
        """
        Visualize word vectors in 2D space with interactive tooltips.
        
        Args:
            words (list): List of words to visualize. If None, visualize all words.
            method (str): Dimensionality reduction method ('pca' or 'tsne')
            perplexity (int): Perplexity parameter for t-SNE
        """
        if self.word_vectors is None:
            raise ValueError("Word vectors not created. Call create_word_vectors first.")
        
        # If words not specified, use top N most frequent words
        if words is None:
            top_words = sorted(self.word_freqs.items(), key=lambda x: x[1], reverse=True)[:100]
            words = [word for word, _ in top_words]
        
        # Filter words that are in vocabulary
        words = [word for word in words if word in self.word_to_idx]
        
        if not words:
            print("No valid words to visualize.")
            return
        
        # Get word vectors for selected words
        word_indices = [self.word_to_idx[word] for word in words]
        vectors = self.word_vectors[word_indices]
        
        # Reduce to 2D for visualization
        if method.lower() == 'pca':
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            title = 'Word Vectors (PCA)'
            # Store the explained variance for explanation
            explained_variance = pca.explained_variance_ratio_
            dimension_explanation = f"PCA dimensions explain {explained_variance[0]*100:.1f}% and {explained_variance[1]*100:.1f}% of variance"
        elif method.lower() == 'tsne':
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            vectors_2d = tsne.fit_transform(vectors)
            title = 'Word Vectors (t-SNE)'
            dimension_explanation = f"t-SNE dimensions represent semantic similarity (perplexity={perplexity})"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # Create an interactive plot
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
        
        # Add word labels
        for i, word in enumerate(words):
            ax.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                        xytext=(5, 2), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        # Add dimension explanation
        ax.text(0.02, 0.98, dimension_explanation,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create a tooltip function
        def on_hover(event):
            if event.inaxes == ax:
                # Find the closest point
                distances = np.sqrt((vectors_2d[:, 0] - event.xdata)**2 + (vectors_2d[:, 1] - event.ydata)**2)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 0.5:  # Only show tooltip if mouse is close enough
                    word = words[closest_idx]
                    word_idx = self.word_to_idx[word]
                    
                    # Get similar words for explanation
                    similar_words = self.get_similar_words(word, top_n=5)
                    similar_words_text = ", ".join([f"{w} ({s:.2f})" for w, s in similar_words])
                    
                    # Get vector dimensions
                    vector = self.word_vectors[word_idx]
                    
                    # Create tooltip text
                    tooltip_text = f"Word: {word}\n"
                    tooltip_text += f"Frequency: {self.word_freqs.get(word, 'N/A')}\n"
                    tooltip_text += f"Similar words: {similar_words_text}\n"
                    tooltip_text += f"Vector dimensions (first 5): {vector[:5]}\n"
                    tooltip_text += f"Vector magnitude: {np.linalg.norm(vector):.3f}"
                    
                    # Update or create annotation
                    try:
                        self.tooltip.set_text(tooltip_text)
                        self.tooltip.xy = (vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1])
                        self.tooltip.set_visible(True)
                    except AttributeError:
                        self.tooltip = ax.annotate(tooltip_text,
                                                xy=(vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1]),
                                                xytext=(20, 20), textcoords="offset points",
                                                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
                    fig.canvas.draw_idle()
                else:
                    try:
                        self.tooltip.set_visible(False)
                        fig.canvas.draw_idle()
                    except AttributeError:
                        pass
        
        # Connect the hover event
        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add instructions
        plt.figtext(0.5, 0.01, "Hover over points to see detailed information about each word",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.show()
    
    def visualize_word_clusters(self, n_clusters=5):
        """
        Visualize word clusters using K-means clustering with interactive tooltips.
        
        Args:
            n_clusters (int): Number of clusters
        """
        if self.word_vectors is None:
            raise ValueError("Word vectors not created. Call create_word_vectors first.")
        
        from sklearn.cluster import KMeans
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.word_vectors)
        
        # Reduce to 2D for visualization
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        vectors_2d = tsne.fit_transform(self.word_vectors)
        
        # Calculate cluster centers in original space
        cluster_centers = kmeans.cluster_centers_
        
        # Find words closest to cluster centers (most representative words)
        representative_words = []
        for cluster_id in range(n_clusters):
            # Calculate distances to cluster center
            distances = np.linalg.norm(self.word_vectors - cluster_centers[cluster_id], axis=1)
            # Get indices of closest words
            closest_indices = np.argsort(distances)[:5]  # Top 5 representative words
            representative_words.append([self.idx_to_word[idx] for idx in closest_indices])
        
        # Create an interactive plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Store scatter plots for each cluster
        scatter_plots = []
        
        # Use different colors for different clusters
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_points = vectors_2d[cluster_mask]
            scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                label=f'Cluster {cluster_id}', alpha=0.7)
            scatter_plots.append((scatter, cluster_mask))
        
        # Add word labels for a subset of words
        top_words_per_cluster = 10
        all_labeled_words = []
        
        for cluster_id in range(n_clusters):
            # Get indices of words in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Get the most frequent words in this cluster
            cluster_words = [(self.idx_to_word[idx], self.word_freqs[self.idx_to_word[idx]])
                            for idx in cluster_indices]
            cluster_words = sorted(cluster_words, key=lambda x: x[1], reverse=True)[:top_words_per_cluster]
            
            # Add labels for these words
            for word, _ in cluster_words:
                idx = self.word_to_idx[word]
                ax.annotate(word, xy=(vectors_2d[idx, 0], vectors_2d[idx, 1]),
                           xytext=(5, 2), textcoords='offset points',
                           fontsize=9, alpha=0.8)
                all_labeled_words.append(word)
        
        # Add cluster explanations
        cluster_explanations = []
        for cluster_id in range(n_clusters):
            explanation = f"Cluster {cluster_id} - Representative words: {', '.join(representative_words[cluster_id])}"
            cluster_explanations.append(explanation)
            
        explanation_text = "\n".join(cluster_explanations)
        ax.text(0.02, 0.98, explanation_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create a tooltip function
        def on_hover(event):
            if event.inaxes == ax:
                # Find the closest point
                distances = np.sqrt((vectors_2d[:, 0] - event.xdata)**2 + (vectors_2d[:, 1] - event.ydata)**2)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 0.5:  # Only show tooltip if mouse is close enough
                    word = self.idx_to_word[closest_idx]
                    cluster_id = clusters[closest_idx]
                    
                    # Find similar words in the same cluster
                    same_cluster_indices = np.where(clusters == cluster_id)[0]
                    same_cluster_words = [self.idx_to_word[idx] for idx in same_cluster_indices
                                         if self.idx_to_word[idx] != word][:10]
                    
                    # Calculate distance to cluster center
                    distance_to_center = np.linalg.norm(self.word_vectors[closest_idx] - cluster_centers[cluster_id])
                    
                    # Get vector dimensions
                    vector = self.word_vectors[closest_idx]
                    
                    # Create tooltip text
                    tooltip_text = f"Word: {word}\n"
                    tooltip_text += f"Cluster: {cluster_id}\n"
                    tooltip_text += f"Frequency: {self.word_freqs.get(word, 'N/A')}\n"
                    tooltip_text += f"Distance to cluster center: {distance_to_center:.3f}\n"
                    tooltip_text += f"Related words in cluster: {', '.join(same_cluster_words[:5])}\n"
                    tooltip_text += f"Vector dimensions (first 5): {vector[:5]}\n"
                    tooltip_text += f"Vector magnitude: {np.linalg.norm(vector):.3f}"
                    
                    # Update or create annotation
                    try:
                        self.tooltip.set_text(tooltip_text)
                        self.tooltip.xy = (vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1])
                        self.tooltip.set_visible(True)
                    except AttributeError:
                        self.tooltip = ax.annotate(tooltip_text,
                                                xy=(vectors_2d[closest_idx, 0], vectors_2d[closest_idx, 1]),
                                                xytext=(20, 20), textcoords="offset points",
                                                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
                    fig.canvas.draw_idle()
                else:
                    try:
                        self.tooltip.set_visible(False)
                        fig.canvas.draw_idle()
                    except AttributeError:
                        pass
        
        # Connect the hover event
        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        
        ax.set_title(f'Word Clusters (K-means, k={n_clusters})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add instructions
        plt.figtext(0.5, 0.01, "Hover over points to see detailed information about each word and its cluster",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.show()
    
    def visualize_word_weights(self):
        """
        Visualize the weights (importance) of words based on frequency and vector magnitude
        with interactive tooltips and explanations.
        """
        if self.word_vectors is None:
            raise ValueError("Word vectors not created. Call create_word_vectors first.")
        
        # Get top 50 most frequent words
        top_words = sorted(self.word_freqs.items(), key=lambda x: x[1], reverse=True)[:50]
        words = [word for word, _ in top_words]
        freqs = [freq for _, freq in top_words]
        
        # Calculate vector magnitudes (L2 norm)
        word_indices = [self.word_to_idx[word] for word in words]
        magnitudes = np.linalg.norm(self.word_vectors[word_indices], axis=1)
        
        # Calculate additional metrics for explanation
        # 1. Variance of each word vector (how distinctive the dimensions are)
        variances = np.var(self.word_vectors[word_indices], axis=1)
        
        # 2. Average similarity to other words (how unique the word is)
        avg_similarities = []
        for idx in word_indices:
            word_vec = self.word_vectors[idx]
            # Compute cosine similarity with all other words
            norm = np.linalg.norm(self.word_vectors, axis=1) * np.linalg.norm(word_vec)
            similarities = np.dot(self.word_vectors, word_vec) / (norm + 1e-8)
            # Average similarity (excluding self)
            avg_sim = (np.sum(similarities) - 1.0) / (len(similarities) - 1)
            avg_similarities.append(avg_sim)
        
        # Create an interactive figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot word frequencies
        bars1 = ax1.barh(range(len(words)), freqs, align='center', color='skyblue')
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels(words)
        ax1.set_title('Word Frequencies')
        ax1.set_xlabel('Frequency')
        
        # Plot vector magnitudes
        bars2 = ax2.barh(range(len(words)), magnitudes, align='center', color='lightgreen')
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words)
        ax2.set_title('Word Vector Magnitudes')
        ax2.set_xlabel('Magnitude')
        
        # Add explanation text
        explanation = """
        Word Weights Explanation:
        
        Frequency: How often a word appears in the corpus.
        - Higher frequency words have more training examples
        - They tend to have more stable representations
        
        Vector Magnitude: The L2 norm of the word vector.
        - Larger magnitudes often indicate more distinctive words
        - Words with similar meanings to many other words may have smaller magnitudes
        - Words with unique contexts tend to have larger magnitudes
        """
        
        fig.text(0.5, 0.01, explanation, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create tooltip functions
        def hover_freq(event):
            if event.inaxes == ax1:
                # Find the closest bar
                for i, bar in enumerate(bars1):
                    if bar.contains(event)[0]:
                        word = words[i]
                        freq = freqs[i]
                        magnitude = magnitudes[i]
                        variance = variances[i]
                        avg_sim = avg_similarities[i]
                        
                        # Create tooltip text
                        tooltip_text = f"Word: {word}\n"
                        tooltip_text += f"Frequency: {freq}\n"
                        tooltip_text += f"Vector magnitude: {magnitude:.3f}\n"
                        tooltip_text += f"Vector variance: {variance:.3f}\n"
                        tooltip_text += f"Avg similarity: {avg_sim:.3f}\n"
                        tooltip_text += f"Percentile rank: {100 * i / len(words):.1f}%"
                        
                        # Update or create annotation
                        try:
                            self.tooltip1.set_text(tooltip_text)
                            self.tooltip1.xy = (freq, i)
                            self.tooltip1.set_visible(True)
                        except AttributeError:
                            self.tooltip1 = ax1.annotate(tooltip_text,
                                                    xy=(freq, i),
                                                    xytext=(20, 0), textcoords="offset points",
                                                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
                        fig.canvas.draw_idle()
                        return
                
                # If not hovering over a bar, hide the tooltip
                try:
                    self.tooltip1.set_visible(False)
                    fig.canvas.draw_idle()
                except AttributeError:
                    pass
        
        def hover_mag(event):
            if event.inaxes == ax2:
                # Find the closest bar
                for i, bar in enumerate(bars2):
                    if bar.contains(event)[0]:
                        word = words[i]
                        freq = freqs[i]
                        magnitude = magnitudes[i]
                        variance = variances[i]
                        avg_sim = avg_similarities[i]
                        
                        # Get similar words
                        similar_words = self.get_similar_words(word, top_n=5)
                        similar_words_text = ", ".join([f"{w}" for w, _ in similar_words])
                        
                        # Create tooltip text
                        tooltip_text = f"Word: {word}\n"
                        tooltip_text += f"Frequency: {freq}\n"
                        tooltip_text += f"Vector magnitude: {magnitude:.3f}\n"
                        tooltip_text += f"Vector variance: {variance:.3f}\n"
                        tooltip_text += f"Avg similarity: {avg_sim:.3f}\n"
                        tooltip_text += f"Similar words: {similar_words_text}"
                        
                        # Update or create annotation
                        try:
                            self.tooltip2.set_text(tooltip_text)
                            self.tooltip2.xy = (magnitude, i)
                            self.tooltip2.set_visible(True)
                        except AttributeError:
                            self.tooltip2 = ax2.annotate(tooltip_text,
                                                    xy=(magnitude, i),
                                                    xytext=(20, 0), textcoords="offset points",
                                                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
                        fig.canvas.draw_idle()
                        return
                
                # If not hovering over a bar, hide the tooltip
                try:
                    self.tooltip2.set_visible(False)
                    fig.canvas.draw_idle()
                except AttributeError:
                    pass
        
        # Connect the hover events
        fig.canvas.mpl_connect("motion_notify_event", hover_freq)
        fig.canvas.mpl_connect("motion_notify_event", hover_mag)
        
        # Add instructions
        plt.figtext(0.5, 0.95, "Hover over bars to see detailed information about each word",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for text
        plt.show()
    
    def explain_word_embeddings(self):
        """
        Provide an explanation of word embeddings and how they work.
        
        Returns:
            str: Explanation text
        """
        explanation = """
        Word Embeddings Explanation:
        --------------------------
        
        Word embeddings are dense vector representations of words that capture semantic meaning.
        Unlike one-hot encodings, embeddings place similar words close together in vector space.
        
        How Word Embeddings Work:
        
        1. Co-occurrence Matrix:
           - We count how often words appear near each other in text
           - This captures the context in which words are used
        
        2. Dimensionality Reduction:
           - The co-occurrence matrix is large and sparse
           - We use techniques like SVD to reduce dimensions while preserving relationships
        
        3. Resulting Embeddings:
           - Each word is represented by a dense vector (e.g., 50-300 dimensions)
           - Similar words have similar vectors
           - Vector arithmetic works: king - man + woman ≈ queen
        
        Applications:
        
        - Semantic similarity: Finding related words
        - Document classification: Converting text to numerical features
        - Machine translation: Mapping words between languages
        - Text generation: Providing semantic context for language models
        
        Visualizing Embeddings:
        
        - We use PCA or t-SNE to reduce embeddings to 2D for visualization
        - Clusters often represent semantic or syntactic word groups
        - Distance between words represents semantic similarity
        """
        
        return explanation

def load_sample_texts():
    """
    Load a sample collection of texts for demonstration.
    
    Returns:
        list: List of text strings
    """
    # Sample texts (simplified for demonstration)
    texts = [
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.",
        "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.",
        "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
        "The Transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing.",
        "A convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery.",
        "Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.",
        "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "Big data is a field that treats ways to analyze, systematically extract information from, or otherwise deal with data sets that are too large or complex.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals.",
        "Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum and first released in 1991.",
        "TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks.",
        "PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.",
        "Scikit-learn is a free software machine learning library for the Python programming language.",
        "Pandas is a software library written for the Python programming language for data manipulation and analysis."
    ]
    
    return texts

def generate_text_dataset(n_samples=100, vocab_size=1000, seq_length=20, n_classes=2):
    """
    Generate a synthetic text classification dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        vocab_size (int): Size of vocabulary
        seq_length (int): Length of each text sequence
        n_classes (int): Number of classes
        
    Returns:
        tuple: (texts, labels) where texts is a list of strings and labels is a numpy array
    """
    # Generate random vocabulary
    vocabulary = [f"word_{i}" for i in range(vocab_size)]
    
    # Generate class-specific word distributions
    class_word_probs = []
    for _ in range(n_classes):
        # Each class has a different distribution over words
        probs = np.random.dirichlet(np.ones(vocab_size) * 0.1)
        class_word_probs.append(probs)
    
    # Generate texts and labels
    texts = []
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Assign a random class
        class_id = i % n_classes
        labels[i] = class_id
        
        # Generate text using the class-specific word distribution
        words = np.random.choice(vocabulary, size=seq_length, p=class_word_probs[class_id])
        text = " ".join(words)
        texts.append(text)
    
    return texts, labels

def texts_to_sequences(texts, word_to_idx, max_length=None):
    """
    Convert texts to sequences of word indices.
    
    Args:
        texts (list): List of text strings
        word_to_idx (dict): Word to index mapping
        max_length (int): Maximum sequence length (pad/truncate)
        
    Returns:
        numpy.ndarray: Sequences of word indices
    """
    sequences = []
    
    for text in texts:
        # Preprocess text
        tokens = text.lower().split()
        
        # Convert to indices
        seq = [word_to_idx.get(token, 0) for token in tokens if token in word_to_idx]
        
        # Pad or truncate if needed
        if max_length is not None:
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [0] * (max_length - len(seq))
        
        sequences.append(seq)
    
    return np.array(sequences)