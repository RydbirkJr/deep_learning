from abc import ABCMeta, abstractmethod


class Network(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def get_batch(self, batch_size):
        raise NotImplementedError()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)