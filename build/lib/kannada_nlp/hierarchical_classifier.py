import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


class HierarchicalClassifier:
    def __init__(self, input_dim, num_coarse_classes, num_fine_classes):
        """
        Initializes the Hierarchical Classifier, which classifies input into two hierarchical categories.

        Args:
        - input_dim: The dimensionality of the input data.
        - num_coarse_classes: The number of coarse-level classes (top-level categories).
        - num_fine_classes: The number of fine-level classes (sub-categories under each coarse class).
        """
        self.input_dim = input_dim
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes

    def build_model(self):
        """
        Builds the hierarchical classification model with two output branches: coarse and fine.

        Returns:
        - model: The hierarchical classification model with two output layers.
        """
        # Input layer
        inputs = Input(shape=(self.input_dim,))

        # Dense layer shared by both branches
        dense1 = Dense(128, activation="relu")(inputs)

        # Coarse output: Predicting high-level categories
        coarse_output = Dense(self.num_coarse_classes, activation="softmax", name="coarse_output")(dense1)

        # Fine output: Predicting sub-categories under the coarse categories
        dense2 = Dense(64, activation="relu")(dense1)  # Another dense layer for fine output
        fine_output = Dense(self.num_fine_classes, activation="softmax", name="fine_output")(dense2)

        # Define the model with both coarse and fine outputs
        model = Model(inputs, outputs=[coarse_output, fine_output])
        return model
