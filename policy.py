from __future__ import division

import numpy as np


class Policy:

    def __init__(self,basis, num_theta, env):
        self.basis_function=basis
        self.actions = range(env.action_space.n)

        self.weights = np.random.uniform(-1.0, 1.0, size=(num_theta,))
        #self.weights = np.zeros((num_theta,))


    def q_value_function(self, state, action ):
        vector_basis = self.basis_function.basisfunc(state, action)
        return np.dot(np.array(vector_basis),self.weights)

    def get_actions(self, state):

        q_state_action=[self.q_value_function(state,a) for a in self.actions]
        q_state_action = np.reshape(q_state_action,[len(q_state_action),1])# convert to column vector

        index = np.argmax(q_state_action)
        q_max = np.max(q_state_action)

        best_actions = [i for i, q in enumerate(q_state_action) if q == q_max]

        return best_actions







