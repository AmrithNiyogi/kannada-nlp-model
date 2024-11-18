import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Embedding

from .attention_layer import AttentionLayer


class BiLSTMModel:
    def __init__(self, vocab_size, embedding_dim, max_seq_length, embedding_matrix=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        # Input layer with the shape of the sequence length
        inputs = Input(shape=(self.max_seq_length,))

        # Embedding layer: Uses pre-trained embedding matrix if provided
        if self.embedding_matrix is not None:
            embedding = Embedding(self.vocab_size, self.embedding_dim,
                                  weights=[self.embedding_matrix], trainable=False)(inputs)
        else:
            embedding = Embedding(self.vocab_size, self.embedding_dim)(inputs)

        # Bi-directional LSTM layer for sequence processing
        lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding)

        # Attention layer to focus on relevant parts of the sequence
        context_vector, attention_weights = AttentionLayer()(lstm)

        # Dropout for regularization to avoid overfitting
        dropout = Dropout(0.5)(context_vector)

        # Output layer with softmax activation for classification
        outputs = Dense(self.vocab_size, activation="softmax")(dropout)

        # Build and return the final model
        model = Model(inputs, outputs)
        return model
