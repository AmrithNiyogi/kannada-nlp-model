import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0):
        """
        Initializes the EarlyStopping callback.

        Args:
        - patience: Number of epochs to wait for improvement before stopping the training.
        - min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')  # Initialize best_loss to infinity
        self.wait = 0  # Counter for the number of epochs without improvement

    def on_epoch_end(self, epoch, logs=None):
        """
        This function is called at the end of each epoch. It monitors the validation loss
        to check if there has been an improvement.

        Args:
        - epoch: The current epoch number.
        - logs: A dictionary of metrics at the end of the epoch, including 'val_loss'.
        """
        current_loss = logs.get('val_loss')  # Get the validation loss for the current epoch
        if current_loss < self.best_loss - self.min_delta:
            # If the loss improved by at least `min_delta`, update best_loss and reset the wait counter
            self.best_loss = current_loss
            self.wait = 0
        else:
            # If no improvement, increment the wait counter
            self.wait += 1
            if self.wait >= self.patience:
                # If the wait counter exceeds patience, stop training
                print(f"Early stopping at epoch {epoch + 1}")
                self.model.stop_training = True
