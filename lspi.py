
import numpy as np
from rbf import RadialBasisFunction
from policy import Policy


class LSPI:

    def __init__(self, env, indim, basisType, alpha, gamma, fancyBasis=False, phibeUpdate=False):

        #print(f"LSPI created with {num_actions} actions, {num_means} bases")

        if basisType == "radial":
            self.basis_function = RadialBasisFunction(indim, 1, fancyBasis)
        else: 
            ValueError()

        self.num_weights = self.basis_function.numWeights
        self.policy = Policy(self.basis_function, self.num_weights, env)
        self.lstdq  = LSTDQ(self.basis_function, gamma, self.policy, alpha, phibeUpdate=phibeUpdate)

        self.stop_criterium= 10**-5
        self.gamma = gamma
        self.alpha=alpha

    def _act(self,state):
        best_actions =  self.policy.get_actions(state)  # TODO: validation for random actions
        return best_actions


    def train( self,  sample,  total_iterations ):

        error = float('inf')
        error_log = []
        num_iteration=0
        eps = 1e-5

        #print "policy weights", self.policy.weights

        while eps < error and num_iteration< total_iterations :
            new_weights = self.lstdq.train_parameter(sample,self.basis_function)

            error = np.linalg.norm((new_weights - self.policy.weights))#difference between current policy and target policy
            error_log.append(error)
            self.policy.weights = self.policy.weights*(1-self.alpha) + (new_weights*self.alpha).ravel() 

            num_iteration += 1

        return self.policy
    


class LSTDQ:
    def __init__(self,basis_function, gamma, policy, alpha, phibeUpdate=False):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = policy
        self.alpha = alpha
        self.phibeUpdate = phibeUpdate

    def train_parameter1 (self, sample, basis_function ): # ORIGINAL UPDATE
        r""" Compute Q value function of current policy
            to obtain the greedy policy
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

            action= self.policy.get_actions(next_states[i])

            phi =      self.basis_function.basisfunc(states[i], actions[i])
            phi_next = self.basis_function.basisfunc(next_states[i], action)

            # ------ABSORBING STATES------#
            #if rewards[i] == -1:
                #phi_next *= 0

            loss = (phi - self.gamma * np.array(phi_next))
            phi  = np.resize(phi, [k, 1])

            loss = np.resize(loss, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
  
    def train_parameter2 (self, sample, basis_function ):
        r""" Compute Q value function of current policy
            to obtain the greedy policy
        """
        tau = 1
        k = basis_function.numWeights

        A=np.zeros([k,k])
        b=np.zeros([k,1])
        np.fill_diagonal(A, 0.1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = (sample[2] / tau).ravel()
        next_states = sample[3]

        for i in range(len(states)):

            # take action from the greedy target policy
            action= self.policy.get_actions(next_states[i])

            phi      = self.basis_function.basisfunc(states[i], actions[i])
            grad_s   = self.basis_function.grad_s(states[i], actions[i])
            grad_a   = self.basis_function.grad_a(states[i], actions[i])

            beta = np.log(self.gamma)/-tau
            A_i = beta * phi - np.dot(grad_s, (next_states[i] - states[i])/tau) + (action-actions[i])/tau * grad_a
            b_i = rewards[i]
            
            phi  = np.reshape(phi, [k, 1])

            A_real = np.matmul(phi, A_i.reshape([1,k]))
            b_real = b_i * phi

            A = A + A_real
            b = b + b_real

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
    
    def train_parameter3 (self, sample, basis_function ):
        """ Compute Q value function of current policy
            to obtain the greedy policy
        """
        delta_t = 0.1215
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
            action= self.policy.get_actions(next_states[i])

            phi      = self.basis_function.basisfunc(states[i], actions[i])
            grad_s   = self.basis_function.grad_s(states[i], actions[i])
            grad_a   = self.basis_function.grad_a(states[i], actions[i])

            beta = np.log(self.gamma)/-delta_t
            A_i = beta * phi - np.dot(grad_s, (next_states[i] - states[i])/delta_t) - (action-actions[i])/delta_t * grad_a
            #phi_next = phi + (action-actions[i])/delta_t*grad_a + np.dot(grad_s, (next_states[i]-states[i])/delta_t)
            #A_i = phi - 0.95 * phi_next 
            b_i = rewards[i] * delta_t
            
            phi  = np.reshape(phi, [k, 1])

            A_real = np.matmul(phi, A_i.reshape([1,k]))
            b_real = b_i * phi

            A = A + A_real
            b = b + b_real

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
    
    def train_parameter(self, sample, basis_function):
        if self.phibeUpdate:
            return self.train_parameter3(sample, basis_function)
        else:
            return self.train_parameter1(sample, basis_function)