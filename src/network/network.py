import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer
from lasagne.nonlinearities import leaky_rectify, softmax


class Network(object):

    def __init__(self, shape, number_of_ouputs):
        # symbolic variables for state, action, and advantage
        self.sym_state = T.tensor4()
        self.sym_action = T.ivector()
        self.sym_advantage = T.fvector()

        # policy network
        l_in = InputLayer(shape=shape)

        l_conv = Conv2DLayer(incoming=l_in, num_filters=10, filter_size=5, pad='same', stride=1)

        l_hid = DenseLayer(incoming=l_conv, num_units=100, nonlinearity=leaky_rectify, name='hiddenlayer1')
        l_out = DenseLayer(incoming=l_hid, num_units=number_of_ouputs, nonlinearity=softmax, name='outputlayer')

        # get network output
        eval_out = lasagne.layers.get_output(l_out, {l_in: self.sym_state}, deterministic=True)

        # get trainable parameters in the network.
        params = lasagne.layers.get_all_params(l_out, trainable=True)

        # get total number of timesteps
        total_timesteps = self.sym_state.shape[0]

        # loss function that we'll differentiate to get the policy gradient
        loss = -T.log(
            eval_out[T.arange(total_timesteps), self.sym_action]
        ).dot(self.sym_advantage) / total_timesteps

        # learning_rate
        learning_rate = T.fscalar()

        # get gradients
        grads = T.grad(loss, params)
        # update function
        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)

        self.f_train = theano.function(
            [
                self.sym_state,
                self.sym_action,
                self.sym_advantage,
                learning_rate
            ],
            loss,
            updates=updates,
            allow_input_downcast=True
        )
        self.f_eval = theano.function([self.sym_state], eval_out, allow_input_downcast=True)

    def train(self, all_states, all_actions, all_advantages, learning_rate):
        return self.f_train(all_states, all_actions, all_advantages, learning_rate)

    def evaluate(self, state):
        return self.f_eval(state)
