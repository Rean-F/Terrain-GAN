import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class SkipGram(keras.Model):
    def __init__(self, input_dim: int, embed_dim: int):
        """skip-gram model

        Args:
            input_dim (int): token-based geo world dim
            embed_dim (int): latent space dim
        """
        super().__init__()
        # encoder
        self.embedding = layers.Embedding(
            input_dim=input_dim,
            output_dim=embed_dim,
            input_length=1
        )
        self.flatten = layers.Flatten()
        # decoder
        self.decoder = layers.Dense(input_dim, activation=None)

    def call(self, inputs: tf.Tensor, training: bool=None, mask: tf.Tensor=None) -> tf.Tensor:
        """inference

        Args:
            x (tf.Tensor): input

        Returns:
            tf.Tensor: output
        """
        x = inputs
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x
