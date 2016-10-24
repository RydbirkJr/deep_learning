import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax


class TFNetwork:

    def __init__(self, input_shape, number_of_outputs):
        height, width, channels = input_shape
        reset_default_graph()

        x_image_pl = tf.placeholder(tf.float32, [None, height, width, channels], name="x_image_pl")
        is_training_pl = tf.placeholder(tf.bool, name="is_training_pl")

        l_conv1_a = self._conv(x_image_pl, 16, (5, 5), scope="l_conv1_a")
        l_pool1 = self._pool(l_conv1_a, scope="l_pool1")

        l_flatten = flatten(l_pool1, scope="flatten")

        features = batch_norm(l_flatten, scope='features_bn')

        l2 = fully_connected(features, num_outputs=256, activation_fn=relu,
                             normalizer_fn=batch_norm, scope="l2")
        l2 = dropout(l2, is_training=is_training_pl, scope="l2_dropout")

        # y_ is a placeholder variable taking on the value of the target batch.
        y = fully_connected(l2, number_of_outputs, activation_fn=softmax, scope="y")

        # add TensorBoard summaries for all variables
        #tf.contrib.layers.summarize_variables()

        clip_norm = 1

        ts_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="targets_pl")
        lr_pl = tf.placeholder(tf.float32, [], name="learning_rate_pl")

        # Predictions is the y from the CNN
        def loss_and_acc(predictions):
            # computing cross entropy per sample
            cross_entropy = -tf.reduce_sum(ts_pl * tf.log(predictions + 1e-10), reduction_indices=[1])
            # averaging over samples
            cost = tf.reduce_mean(cross_entropy)

            # if you want regularization
            reg_scale = 0.0002
            regularize = tf.contrib.layers.l1_regularizer(reg_scale)
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            cost += tf.contrib.layers.apply_regularization(regularize, weights_list=params)

            # calculate accuracy
            argmax_y = tf.to_int32(tf.argmax(predictions, dimension=1))
            argmax_t = tf.to_int32(tf.argmax(ts_pl, dimension=1))
            correct = tf.to_float(tf.equal(argmax_y, argmax_t))
            accuracy = tf.reduce_mean(correct)
            return cost, accuracy, argmax_y


        # loss, accuracy and prediction
        loss_valid, accuracy_valid, _ = loss_and_acc(y)

        # defining our optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)

        # applying the gradients
        grads_and_vars = optimizer.compute_gradients(loss_valid)
        gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
        clipped_gradients, global_norm = (
            tf.clip_by_global_norm(gradients, clip_norm))
        clipped_grads_and_vars = zip(clipped_gradients, variables)

        # make training op for applying the gradients
        train_op = optimizer.apply_gradients(clipped_grads_and_vars)

        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))

        sess.run(tf.initialize_all_variables())




    # easy to use pool function
    @staticmethod
    def _pool(l_in, scope, kernel_size=(3, 3)):
        return max_pool2d(l_in, kernel_size=kernel_size, scope=scope)  # (3, 3) has shown to work better than (2, 2)


    @staticmethod
    def _conv(l_in, num_outputs, kernel_size, scope, stride=1):
        return convolution2d(l_in, num_outputs=num_outputs, kernel_size=kernel_size,
                             stride=stride, normalizer_fn=batch_norm, scope=scope)


