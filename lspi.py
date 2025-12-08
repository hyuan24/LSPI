
import numpy as np
from rbf import RadialBasisFunction, PolynomialBasisFunction
from policy import Policy

class LSPI:

    def __init__(self, env, indim, basisType, alpha = 1.0, gamma=0.95):

        #print(f"LSPI created with {num_actions} actions, {num_means} bases")

        self.num_weights = 30
        if basisType == "radial":
            self.basis_function = RadialBasisFunction(indim, 1)
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
        errors = []
        weights_history = []
        num_iteration=0
        eps = 1e-5

        while eps < error and num_iteration< total_iterations :
            weights_history.append(self.policy.weights)
            new_weights = self.lstdq.train_parameter(sample,self.basis_function)

            error = np.linalg.norm((new_weights - self.policy.weights))#difference between current policy and target policy
            errors.append(error)
            self.policy.weights = new_weights

            num_iteration += 1

        return self.policy
    


class LSTDQ:
    def __init__(self,basis_function, gamma, policy, alpha, normalize=True):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = policy
        self.alpha = alpha
        self.normalize = normalize # Normalize theta dot

    def train_parameter (self, sample, basis_function):
        r""" Compute Q value function of current policy
            to obtain the gready policy
        """
        tau = 0.1225
        k = basis_function.numWeights

        A=np.zeros([k,k])
        b=np.zeros([k,1])
        np.fill_diagonal(A, 1e-1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2] * tau
        next_states = sample[3]

        if self.normalize:
            angle_dots = [state[1] for state in states]
            ad_mean, ad_std = np.mean(angle_dots), np.std(angle_dots)
            states = np.array([[state[0], (state[1]-ad_mean)/ad_std] for state in states])

            n_angle_dots = [nstate[1] for nstate in next_states]
            n_ad_mean, n_ad_std = np.mean(n_angle_dots), np.std(n_angle_dots)
            next_states = np.array([[nstate[0], (nstate[1]-n_ad_mean)/n_ad_std] for nstate in next_states])


        for i in range(len(states)):

            # take action from the greedy target policy
            action= self.policy.get_actions(next_states[i])

            phi      = self.basis_function.basisfunc(states[i], actions[i])
            #phi_next = self.basis_function.basisfunc(next_states[i], action)
            grad_s   = self.basis_function.grad_s(states[i], actions[i])
            grad_a   = self.basis_function.grad_a(states[i], actions[i])

            # ------ABSORBING STATES------#
            #if rewards[i] == -1:
                #phi_next *= 0

            beta = np.log(self.gamma)/-tau
            A_i = beta * phi - np.dot(grad_s, (next_states[i] - states[i])/tau) + (action-actions[i])/tau * grad_a
            b_i = rewards[i]
            
            phi  = np.reshape(phi, [k, 1])

            A_real = np.matmul(phi, A_i.reshape([1,k]))
            b_real = b_i * phi
            #loss = (phi - self.gamma * np.array(phi_next))
            #phi  = np.resize(phi, [k, 1])

            #loss = np.resize(loss, [1, len(loss)])

            A = A + self.alpha * A_real
            b = b + self.alpha * b_real

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
  

  
