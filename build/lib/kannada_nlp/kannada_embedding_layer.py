# kannada_nlp/kannada_embedding_layer.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *


class KannadaEmbeddingLayer(Layer):
    """
    Custom embedding layer for Kannada text data.
    Allows integration of pre-trained word embeddings.
    """

    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, trainable=True, **kwargs):
        super(KannadaEmbeddingLayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable

    def build(self, input_shape):
        """Initialize the weights for the embedding layer."""
        if self.embedding_matrix is not None:
            # If embedding matrix is provided (pre-trained), use it
            self.embedding_layer = Embedding(input_dim=self.vocab_size,
                                             output_dim=self.embedding_dim,
                                             weights=[self.embedding_matrix],
                                             trainable=self.trainable)
        else:
            # Otherwise, initialize embedding layer randomly
            self.embedding_layer = Embedding(input_dim=self.vocab_size,
                                             output_dim=self.embedding_dim,
                                             trainable=self.trainable)

        super(KannadaEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        """Apply embedding to the inputs."""
        return self.embedding_layer(inputs)

    def get_config(self):
        """Return configuration for the custom layer."""
        config = super(KannadaEmbeddingLayer, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "embedding_matrix": self.embedding_matrix,
            "trainable": self.trainable
        })
        return config


def create_embedding_matrix(tokenizer, embedding_file, embedding_dim=300):
    """
    Create the embedding matrix using a pre-trained embedding file.
    Tokenizer's word index is used to map words to their corresponding embeddings.

    Parameters:
    - tokenizer: A fitted Tokenizer object.
    - embedding_file: Path to pre-trained word embeddings (e.g., FastText, GloVe, Word2Vec).
    - embedding_dim: The dimension of the embedding vectors.

    Returns:
    - embedding_matrix: A numpy array representing the embedding matrix.
    """
    # Initialize the embedding matrix with zeros
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

    # Load the pre-trained word embeddings (assuming it's a text file)
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')

            # If the word is in tokenizer's word index, update its embedding
            if word in tokenizer.word_index:
                index = tokenizer.word_index[word]
                embedding_matrix[index] = vector

    return embedding_matrix


# Example Usage
if __name__ == "__main__":
    # Sample tokenizer (you would fit it with your dataset)
    tokenizer = Tokenizer()
    sample_texts = ["ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ", "ನಾನು ಸಂಶೋಧನೆ ಮಾಡುತ್ತಿದ್ದೇನೆ"]
    tokenizer.fit_on_texts(sample_texts)

    # Path to pre-trained Kannada word embeddings (change this path to your embeddings)
    embedding_file = 'path_to_pretrained_embeddings.vec'

    # Create the embedding matrix using pre-trained word embeddings
    embedding_matrix = create_embedding_matrix(tokenizer, embedding_file)

    # Instantiate the custom embedding layer
    embedding_layer = KannadaEmbeddingLayer(vocab_size=len(tokenizer.word_index) + 1,
                                            embedding_dim=300,  # Adjust dimension based on the pre-trained embeddings
                                            embedding_matrix=embedding_matrix,
                                            trainable=False)

    # Now you can use this embedding layer in your model
    print("Custom Kannada Embedding Layer Created!")
