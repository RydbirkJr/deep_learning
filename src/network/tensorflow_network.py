import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
import numpy as np


class TFNetwork:
    def __init__(self, input_shape, number_of_outputs):
        self.number_of_outputs = number_of_outputs
        # height, width, channels = input_shape
        reset_default_graph()

        x_image_pl = tf.placeholder(tf.float32, input_shape, name='x_image_pl')
        actions_pl = tf.placeholder(tf.float32, name='actions_pl')
        advantages_pl = tf.placeholder(tf.float32, name='advantages_pl')
        is_training_pl = tf.placeholder(tf.bool, name='is_training_pl')
        learning_rate_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')

        # add TensorBoard summaries for all variables
        # tf.contrib.layers.summarize_variables()

        self.y = self._get_network(number_of_outputs, x_image_pl, is_training_pl)

        self.training = self._policy_gradient(self.y, actions_pl, advantages_pl, learning_rate_pl)

        self.x_image_pl = x_image_pl
        self.actions_pl = actions_pl
        self.advantages_pl = advantages_pl
        self.is_training_pl = is_training_pl
        self.learning_rate_pl = learning_rate_pl

        # gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

        # initialize the Session
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
        tf.device('/cpu:0')
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # easy to use pool function
    @staticmethod
    def _pool(l_in, scope, kernel_size=(3, 3)):
        return max_pool2d(l_in, kernel_size=kernel_size, scope=scope)  # (3, 3) has shown to work better than (2, 2)

    @staticmethod
    def _conv(l_in, num_outputs, kernel_size, scope, stride=1):
        return convolution2d(l_in, num_outputs=num_outputs, kernel_size=kernel_size,
                             stride=stride, normalizer_fn=batch_norm, scope=scope)

    def _get_network(self, number_of_outputs, input_pl, is_training_pl):
        #l_conv1_a = self._conv(input_pl, 16, (5, 5), scope="l_conv1_a")
        #l_pool1 = self._pool(l_conv1_a, scope="l_pool1")
        #l_flatten = flatten(l_pool1, scope="flatten")
        #features = batch_norm(l_flatten, scope='features_bn')
        #l2 = fully_connected(features, num_outputs=256, activation_fn=relu,
        #                     normalizer_fn=batch_norm, scope='l2')
        #l2 = dropout(l2, is_training=is_training_pl, scope="l2_dropout")

        # y_ is a placeholder variable taking on the value of the target batch.
        # Y = probabilities
        #y = fully_connected(l2, number_of_outputs, activation_fn=softmax, scope="y")

        l_in = fully_connected(input_pl, activation_fn=relu,
                             normalizer_fn=batch_norm, scope='l2')
        l_hid = fully_connected(l_in, num)

        return y

    def _policy_gradient(self, y, actions_pl, advantages_pl, learning_rate_pl):
        # tvars = tf.trainable_variables()

        # input_y is actions
        # His advantages does not have shape assigned

        # good_prob = tf.reduce_sum(tf.mul(y, actions_pl), reduction_indices=[1])

        # computing cross entropy per sample

        # Try out tf.nn.softmax_something_logits
        cross_entropy = -tf.reduce_sum((tf.log(actions_pl - y)) * advantages_pl, reduction_indices=[1])

        # averaging over samples
        self.cost = tf.reduce_mean(cross_entropy)
        # newGrads = tf.gradients(cost, tvars)

        # If regularization is wanted
        # reg_scale = 0.0002
        # regularize = tf.contrib.layers.l1_regularizer(reg_scale)
        # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # cost += tf.contrib.layers.apply_regularization(regularize, weights_list=params)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
        # return optimizer.minimize(cost)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        return optimizer.minimize(self.cost, global_step=global_step)

    #        clip_norm = 1

    # applying the gradients
    # grads_and_vars = optimizer.compute_gradients(cost)
    # gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
    # clipped_gradients, global_norm = (
    #     tf.clip_by_global_norm(gradients, clip_norm))
    # clipped_grads_and_vars = zip(clipped_gradients, variables)

    # make training op for applying the gradients
    # return optimizer.apply_gradients(clipped_grads_and_vars)

    # Train the network
    def train(self, all_states, all_actions, all_advantages, learning_rate):
        # advantages_vector = np.expand_dims(all_advantages, axis=1)

        all_actions = self.onehot(all_actions, self.number_of_outputs)
        all_advantages = self.onehot_advantages(all_advantages, self.number_of_outputs, all_actions)

        print 'all_actions: ', all_actions.shape
        print all_actions
        print 'all_advantages shape: ', all_advantages.shape
        print all_advantages

        # print 'all_advantages shape: ', advantages_vector.shape
        feed_dict_train = {
            self.x_image_pl: all_states,
            self.advantages_pl: all_advantages,
            self.learning_rate_pl: learning_rate,
            self.is_training_pl: True,
            self.actions_pl: all_actions
        }

        # No idea how to implement all_actions and all_advantages - is it a part of shape?
        res = self.sess.run([self.training, self.cost], feed_dict=feed_dict_train)
        print res
        return res[1]

    # Evaluate the network
    def evaluate(self, state):
        return self.sess.run(self.y, feed_dict={self.is_training_pl: False, self.x_image_pl: state})

    def onehot(self, t, num_classes):
        out = np.zeros((t.shape[0], num_classes))
        for row, col in enumerate(t):
            out[row, col] = 1
        return out

    # Assume that mask has the shape that t will be transformed into
    def onehot_advantages(self, t, num_classes, mask):
        out = np.zeros((t.shape[0], num_classes))
        for row, val in enumerate(t):
            for col in range(num_classes):
                # multiplies with the mask: if 1, the val is writen to the out matrix
                out[row, col] = mask[row, col] * val
        return out

