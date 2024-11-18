import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout

class EnsembleModel:
    def __init__(self, lstm_model, cnn_model):
        """
        Initializes the Ensemble Model which combines LSTM and CNN models.

        Args:
        - lstm_model: A trained or untrained BiLSTM model.
        - cnn_model: A trained or untrained CNN model.
        """
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model

    def build_model(self):
        """
        Builds the ensemble model by combining the outputs of both the LSTM and CNN models.

        Returns:
        - model: The combined ensemble model.
        """
        # Get the outputs of both models
        lstm_output = self.lstm_model.output
        cnn_output = self.cnn_model.output

        # Concatenate the outputs of LSTM and CNN
        merged = Concatenate()([lstm_output, cnn_output])

        # Apply dropout for regularization
        dropout = Dropout(0.5)(merged)

        # Define the final output layer with softmax activation for classification
        outputs = Dense(self.lstm_model.output_shape[-1], activation="softmax")(dropout)

        # Build the final model with both LSTM and CNN inputs
        model = Model(inputs=[self.lstm_model.input, self.cnn_model.input], outputs=outputs)
        return model
