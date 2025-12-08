from __future__ import division

import numpy as np


class Policy:

    def __init__(self,basis, num_theta, env, eps_greedy=1):
        self.basis_function=basis
        self.actions = range(env.action_space.n)
        self.env = env
        #self.weights = np.random.normal(0, 1, size=(num_theta,))
        self.weights = np.zeros((num_theta,)) # MODIFIED
        self.eps_greedy  = eps_greedy

    def q_value_function(self, state, action ):
        vector_basis = self.basis_function.basisfunc(state, action)
        return np.dot(np.array(vector_basis),self.weights)

    def get_actions(self, state):

        q_state_action=[self.q_value_function(state,a) for a in self.actions]
        q_state_action = np.reshape(q_state_action,[len(q_state_action),1])# convert to column vector

        index = np.argmax(q_state_action)
        q_max = np.max(q_state_action)

        best_actions = [i for i, q in enumerate(q_state_action) if q == q_max]
        if np.random.uniform(0,1) <= self.eps_greedy:
            return np.random.choice(best_actions)
        else:
            return self.env.action_space.sample()







