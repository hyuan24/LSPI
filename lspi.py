
import numpy as np
from rbf import RadialBasisFunction, PolynomialBasisFunction
from policy import Policy


class LSPI:

    def __init__(self, num_actions, num_means, env, indim, basisType, alpha = 1.0, gamma=0.95):

        #print(f"LSPI created with {num_actions} actions, {num_means} bases")

        self.num_weights = num_means*num_actions
        if basisType == "radial":
            self.basis_function = RadialBasisFunction(indim, 1, self.num_weights)
        elif basisType == "poly":
            self.basis_function = PolynomialBasisFunction(indim, 1)
        else: 
            ValueError()

        self.policy = Policy(self.basis_function, self.num_weights, env)
        self.lstdq  = LSTDQ(self.basis_function, gamma, self.policy, alpha)

        self.stop_criterium= 10**-5
        self.gamma = gamma


    def _act(self,state):
        best_actions =  self.policy.get_actions(state)  # TODO: validation for random actions
        return best_actions[0]


    def train( self,  sample,  total_iterations ):

        error = float('inf')
        num_iteration=0
        eps = 1e-5

        #print "policy weights", self.policy.weights

        while eps < error and num_iteration< total_iterations :
            new_weights = self.lstdq.train_parameter(sample,self.basis_function)

            error = np.linalg.norm((new_weights - self.policy.weights))#difference between current policy and target policy
            self.policy.weights = new_weights

            num_iteration += 1
            #print(f"ITERATION {num_iteration} COMPLETE")

        return self.policy
    


class LSTDQ:
    def __init__(self,basis_function, gamma, policy, alpha):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = policy
        self.alpha = alpha

    def train_parameter (self, sample, basis_function ):
        r""" Compute Q value function of current policy
            to obtain the gready policy
        """
        k = basis_function.numWeights

        A=np.zeros([k,k])
        b=np.zeros([k,1])
        np.fill_diagonal(A, 0.1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]

        for i in range(len(states)):

            # take action from the greedy target policy

            action= self.policy.get_actions(next_states[i])[0]

            phi =      self.basis_function.basisfunc(states[i], actions[i])
            phi_next = self.basis_function.basisfunc(next_states[i], action)

            # ------ABSORBING STATES------#
            #if rewards[i] == -1:
                #phi_next *= 0

            loss = (phi - self.gamma * np.array(phi_next))
            phi  = np.resize(phi, [k, 1])

            loss = np.resize(loss, [1, len(loss)])

            A = A + self.alpha * np.dot(phi, loss)
            b = b + self.alpha * (phi * rewards[i])

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
  