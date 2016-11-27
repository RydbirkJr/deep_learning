import lasagne
import theano
import theano.tensor as T
from lasagne.init import Constant, HeUniform
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, get_output
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid, tanh, elu, rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop, adam
import numpy as np


class Network(object):
    def __init__(self, resolution, number_of_outputs, cropping):
        # symbolic variables for state, action, and advantage
        self.sym_state = T.tensor4()
        self.sym_action = T.vector("Actions", dtype="int32")
        self.sym_advantage = T.vector("Advantages", dtype="int32")
        self.sym_r = T.vector()
        self.sym_q2 = T.vector()
        self.shape = (None, 3,
                      (resolution[0] - cropping[0] - cropping[1]),
                      (resolution[1] - cropping[2] - cropping[3]))

        self.cropping = cropping

        # Same policy network as Deep Q
        l_in = InputLayer(shape=self.shape, input_var=self.sym_state)
        l_conv1 = Conv2DLayer(l_in, num_filters=16, filter_size=[8, 8], nonlinearity=rectify, stride=4)
        l_conv2 = Conv2DLayer(l_conv1, num_filters=32, filter_size=[4, 4], nonlinearity=rectify, stride=2)
        l_hid1 = DenseLayer(l_conv2, num_units=256, nonlinearity=rectify)
        self.l_out = DenseLayer(incoming=l_hid1, W=Constant(1), num_units=number_of_outputs, nonlinearity=softmax,
                           name='outputlayer')

        # policy network
        # l_in = InputLayer(shape=shape, input_var=self.sym_state)
        #
        # l_conv1 = Conv2DLayer(incoming=l_in, num_filters=10, filter_size=6, stride=2, nonlinearity=rectify,
        #                       W=Constant(0.000001))
        # l_conv2 = Conv2DLayer(incoming=l_conv1, num_filters=20, filter_size=3, stride=1,
        #                       nonlinearity=rectify, W=Constant(1.0))
        # l_hid = DenseLayer(incoming=l_conv2, num_units=100, W=Constant(1.0), nonlinearity=rectify, name='hiddenlayer1')
        # l_out = DenseLayer(incoming=l_hid, W=Constant(1), num_units=number_of_ouputs, nonlinearity=softmax,
        #                    name='outputlayer')

        # get network output
        eval_out = lasagne.layers.get_output(self.l_out, {l_in: self.sym_state}, deterministic=True)

        # get total number of timesteps
        total_timesteps = self.sym_state.shape[0]

        # loss function that we'll differentiate to get the policy gradient
        loss = -T.log(eval_out[T.arange(total_timesteps), self.sym_action]).dot(self.sym_advantage) / total_timesteps

        # learning_rate
        learning_rate = T.fscalar()


        # get trainable parameters in the network.
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)

        # get gradients
        grads = T.grad(loss, params)

        # update function
        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)

        print "Compiling the network ..."
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
        print "Network compiled."

    def train(self, all_states, all_actions, all_advantages, learning_rate):
        #print np.array(all_states).shape
        #print np.array(all_actions).shape
        #print np.array(all_advantages).shape
        #print np.array(learning_rate).shape
        s1 = all_states
        q2 = all_actions
        a = all_advantages
        r = learning_rate
        #print "s1 shape 0:",s1[0].shape[0]
        #print "len a:",len(a)
        #print "T shape a:", (np.transpose(a)).shape
        #print q2
        #all_advantages = np.transpose(a).astype(np.int32)
        #print all_advantages.shape
        #return self.fn_learn(all_states, all_actions, all_advantages, learning_rate)
        return self.f_train(all_states, all_actions, all_advantages, learning_rate)

    def evaluate(self, state):
        #e = self.fn_get_best_action(state)
        #return e
        return self.f_eval(state)
