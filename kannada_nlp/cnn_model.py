import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Embedding


class CNNModel:
    def __init__(self, vocab_size, embedding_dim, max_seq_length, embedding_matrix=None):
        """
        Initializes the CNN model with the necessary parameters.

        Args:
        - vocab_size: The size of the vocabulary (number of unique tokens).
        - embedding_dim: The dimensionality of the embedding vectors.
        - max_seq_length: The maximum sequence length for input data.
        - embedding_matrix: Pretrained embedding matrix (optional).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        """
        Builds the CNN model with embedding, convolutional, pooling, dropout, and dense layers.

        Returns:
        - model: The constructed Keras model.
        """
        inputs = Input(shape=(self.max_seq_length,))

        # Embedding layer
        if self.embedding_matrix is not None:
            embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix],
                                  trainable=False)(inputs)
        else:
            embedding = Embedding(self.vocab_size, self.embedding_dim)(inputs)

        # Convolutional layer
        conv = Conv1D(128, kernel_size=5, activation="relu")(embedding)

        # Max pooling layer
        pooling = GlobalMaxPooling1D()(conv)

        # Dropout layer to prevent overfitting
        dropout = Dropout(0.5)(pooling)

        # Output layer (softmax for classification)
        outputs = Dense(self.vocab_size, activation="softmax")(dropout)

        # Compile the model
        model = Model(inputs, outputs)
        return model
