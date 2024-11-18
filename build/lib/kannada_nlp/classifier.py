import tensorflow as tf


class Classifier:
    def __init__(self, model, loss="binary_crossentropy", optimizer="adam", metrics=None):
        """
        Initializes the classifier with a given model, loss function, optimizer, and evaluation metrics.

        Args:
        - model: The Keras model to be trained.
        - loss: The loss function for training the model.
        - optimizer: The optimizer used for training.
        - metrics: A list of evaluation metrics.
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics or ["accuracy"]

    def compile_model(self):
        """
        Compiles the model with the specified loss function, optimizer, and metrics.
        """
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )

    def train(self, train_data, train_labels, validation_data=None, batch_size=32, epochs=10, callbacks=None):
        """
        Trains the model on the provided training data and labels.

        Args:
        - train_data: The data used for training.
        - train_labels: The corresponding labels for training.
        - validation_data: The data for validation (optional).
        - batch_size: The number of samples per batch.
        - epochs: The number of training epochs.
        - callbacks: A list of callbacks to apply during training (optional).

        Returns:
        - history: A history object containing training metrics.
        """
        history = self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def evaluate(self, test_data, test_labels):
        """
        Evaluates the model on the provided test data and labels.

        Args:
        - test_data: The data used for testing.
        - test_labels: The corresponding labels for testing.

        Returns:
        - The evaluation results (loss and metrics).
        """
        return self.model.evaluate(test_data, test_labels)

    def predict(self, data):
        """
        Makes predictions using the trained model.

        Args:
        - data: The data to be predicted.

        Returns:
        - The model's predictions.
        """
        return self.model.predict(data)

    def save_model(self, path):
        """
        Saves the trained model to the specified path.

        Args:
        - path: The path where the model should be saved.
        """
        self.model.save(path)

    @staticmethod
    def load_model(path):
        """
        Loads a previously saved model from the specified path.

        Args:
        - path: The path from which to load the model.

        Returns:
        - A Keras model object.
        """
        return tf.keras.models.load_model(path)
