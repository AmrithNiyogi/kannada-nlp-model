import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize weights and bias for the attention mechanism
        self.W = self.add_weight(name="attention_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="attention_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="context_vector",
                                 shape=(input_shape[-1],),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention scores
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)

        # Compute the weighted sum of the input based on attention weights
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
