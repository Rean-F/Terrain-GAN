import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D, LeakyReLU, ZeroPadding3D
from tensorflow.keras.layers import Add


class InstanceNorm(Layer):
    """Instance Normalization
    """
    def __init__(self, epsilon: float=1e-6):
        """initialize

        Args:
            epsilon (float, optional): small float added to variance to avoid dividing by zero. defaults to 1e-6.
        """
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape: tf.TensorShape):
        """scale(gamma) and center(beta) to learn

        Args:
            input_shape (tf.TensorShape): [batch_size, h, w, channels]
        """
        # center
        self.beta = tf.Variable(tf.zeros([input_shape[4]]))
        # scale
        self.gamma = tf.Variable(tf.ones([input_shape[4]]))

    def call(self, inputs):
        # calculate mean and variance of each channel
        mean, var = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))

        return self.gamma * x + self.beta

class ConvBlock3D(Layer):
    def __init__(self, num_filters: int):
        """initialize

        Args:
            num_filters (int): conv filters
        """
        super().__init__()
        self.num_filters = num_filters
        self.conv_initializer = tf.random_normal_initializer(0., 0.02)
        self.batch_initializer = tf.random_normal_initializer(1.0, 0.02)
        self.conv_3d = Conv3D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="valid",
            use_bias=True,
            kernel_initializer=self.conv_initializer
        )
        # use instance norm which is better than batch norm in generated model
        # self.batch_norm = BatchNormalization(training=False, axis=-1, gamma_initializer=self.batch_initializer)
        self.instance_norm = InstanceNorm()
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.conv_3d(x)
        # x = self.batch_norm(x)
        x = self.instance_norm(x)
        x = self.leaky_relu(x)

        return x


class Generator(Model):
    """Generator
    """
    def __init__(self, channels: int, num_filters: int, name: str="Generator"):
        """[summary]

        Args:
            channels (int): generated output channels, same to the embedding dims.
            num_filters (int): conv block filters
            name (str, optional): tensor name. Defaults to "Generator".
        """
        super().__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        # padding_size = 5, because of five conv blocks totally
        self.padding = ZeroPadding3D(5)
        self.head = ConvBlock3D(num_filters)
        self.conv_block1 = ConvBlock3D(num_filters)
        self.conv_block2 = ConvBlock3D(num_filters)
        self.conv_block3 = ConvBlock3D(num_filters)
        # don't use tanh for limiting value to [-1, 1]
        self.tail = Conv3D(
            filters=channels,
            kernel_size=3,
            strides=1,
            padding="valid",
            # activation="tanh",
            kernel_initializer=self.initializer
        )

    def call(self, prev, noise):
        prev_pad = self.padding(prev)
        noise_pad = self.padding(noise)
        x = Add()([prev_pad, noise_pad])
        x = self.head(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.tail(x)
        x = Add()([x, prev])

        return x


class Discriminator(Model):
    """Discriminator
    """
    def __init__(self, num_filters: int, name="Discriminator"):
        """[summary]

        Args:
            num_filters (int): conv block filters
            name (str, optional): tensor name. Defaults to "Discriminator".
        """
        super().__init__()
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        # add padding, because y value (down to up) is usally small
        self.padding = ZeroPadding3D(5)
        self.head = ConvBlock3D(num_filters)
        self.conv_block1 = ConvBlock3D(num_filters)
        self.conv_block2 = ConvBlock3D(num_filters)
        self.conv_block3 = ConvBlock3D(num_filters)
        self.tail = Conv3D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding="valid",
            kernel_initializer=self.initializer
        )

    def call(self, x):
        x = self.padding(x)
        x = self.head(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.tail(x)

        return x