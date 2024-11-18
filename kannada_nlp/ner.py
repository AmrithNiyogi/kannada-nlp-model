import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow_addons.text import crf_log_likelihood, viterbi_decode

class KannadaNER:
    def __init__(self, vocab_size, embedding_dim, num_tags, max_seq_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_tags = num_tags
        self.max_seq_length = max_seq_length
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the BiLSTM-CRF model.
        """
        input_layer = tf.keras.Input(shape=(self.max_seq_length,), dtype="int32")
        embeddings = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(input_layer)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
        dense_out = TimeDistributed(Dense(self.num_tags))(lstm_out)

        model = tf.keras.Model(inputs=input_layer, outputs=dense_out)
        model.compile(optimizer="adam", loss=self._crf_loss)
        return model

    def _crf_loss(self, y_true, y_pred):
        """
        CRF loss function.
        """
        log_likelihood, _ = crf_log_likelihood(y_pred, y_true, np.array([self.max_seq_length]))
        return -tf.reduce_mean(log_likelihood)

    def train(self, train_data, train_labels, batch_size=32, epochs=10):
        """
        Train the NER model.
        """
        self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)

    def predict(self, input_sequence):
        """
        Predict entities for an input sequence.
        """
        logits = self.model(input_sequence)
        sequence_length = np.array([len(input_sequence[0])])
        _, tags = viterbi_decode(logits, sequence_length)
        return tags

    def save_model(self, path):
        """
        Save the trained model.
        """
        self.model.save(path)

    @staticmethod
    def load_model(path):
        """
        Load a saved model.
        """
        return tf.keras.models.load_model(path)
