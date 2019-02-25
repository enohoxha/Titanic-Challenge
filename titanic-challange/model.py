import tensorflow as tf


class TitanicModel:

    def __init__(self):
        self.x_train = tf.placeholder(tf.float64, shape=[None, 5], name="x")
        self.y_train = tf.placeholder(tf.float64, shape=[None, 2], name="eno")
        self.hold_prop = tf.placeholder(tf.float32)


    @staticmethod
    def init_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64))

    @staticmethod
    def init_bias(shape):
        return tf.constant(0.1, tf.float64, shape)

    def fully_connected_layer(self, input_layer, size):
        input_layer = tf.cast(input_layer, tf.float64)
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        x = tf.matmul(input_layer, W) + b

        norm = tf.layers.batch_normalization(x, training=True)
        return norm
