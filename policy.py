from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


class Policy:

    def __init__(self,basis, num_theta, env, tau):
        self.basis_function=basis
        self.actions = range(env.action_space.n)
        self.state_dim = env.observation_space.shape
        #self.weights = np.random.normal(0, 0.01, size=(num_theta,))
        self.weights = np.zeros(shape=(num_theta,))
        self.tau = tau

    def q_value_function(self, state, action ):
        vector_basis = self.basis_function.basisfunc(state, action)
        return np.dot(np.array(vector_basis),self.weights)

    def softmax(self, x):
        x = np.array(x)
        e_x = np.exp((x-np.max(x))/self.tau)
        return e_x / e_x.sum()

    def get_actions(self, state):

        q_state_action=[self.q_value_function(state,a) for a in self.actions]

        q_max = np.max(q_state_action)

        best_actions = [i for i, q in enumerate(q_state_action) if q == q_max]

        return np.random.choice(best_actions)
    
    def softmax_get_action(self, state):
        q_state_action=[self.q_value_function(state,a) for a in self.actions]
        q_softmax = self.softmax(q_state_action)
        return np.dot(self.actions, q_softmax)
    
    def grad_s_logit(self, action, grad_s_list):
        return  1/self.tau * np.dot(grad_s_list[action].T, self.weights)

    def softmax_pi_grad(self, action, pis, summation_b, logit_list):
    
        pi_a_s = pis[action]

        return pi_a_s * (logit_list[action] - summation_b) 

    def plot_pi_grad(self, numTicks=100):
        numTicks=numTicks
        angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
        angles_dot = np.linspace(-5, 5, numTicks)
        pi_grad_graph = np.zeros([numTicks,numTicks])

        for i, a in enumerate(angles):
            for j, ad in enumerate(angles_dot):
                state = [a,ad] 
                pi_grad_graph[i,j] = self.softmax_pi_grad(state, 1)[0]
    
        plt.imshow(pi_grad_graph, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
    
        plt.colorbar()
        plt.show()
    
if __name__ == '__main__':
    from rbf import RadialBasisFunction
    from env2 import ModifiedCartPoleEnv
    import matplotlib.pyplot as plt

    rbf = RadialBasisFunction(2, 1, True)
    env = ModifiedCartPoleEnv("dense")
    policy = Policy(rbf, 40, env, 1)

    numTicks=100
    angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
    angles_dot = np.linspace(-5, 5, numTicks)
    pi_grad_graph = np.zeros([numTicks,numTicks])

    for i, a in enumerate(angles):
        for j, ad in enumerate(angles_dot):
            state = [a,ad] 
            pi_grad_graph[i,j] = policy.softmax_pi_grad(state, 0)[0]
    
    plt.imshow(pi_grad_graph, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
       
    #plt.set_xlabel("theta")
    #plt.set_ylabel("theta_dot")
    
    plt.colorbar()
    plt.show()
                
   




