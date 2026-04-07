
import numpy as np
from rbf import RadialBasisFunction
from policy import Policy

class LSPI:

    def __init__(self, env, indim, basisType, alpha, gamma, tau, fancyBasis=False, phibeUpdate=False):

        #print(f"LSPI created with {num_actions} actions, {num_means} bases")

        if basisType == "radial":
            self.basis_function = RadialBasisFunction(indim, 1, fancyBasis)
        else: 
            ValueError()

        self.num_weights = self.basis_function.numWeights
        self.policy = Policy(self.basis_function, self.num_weights, env, tau)
        self.lstdq  = LSTDQ(self.basis_function, gamma, self.policy, alpha, tau, phibeUpdate=phibeUpdate)

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
    def __init__(self,basis_function, gamma, policy, alpha, tau, phibeUpdate=False):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = policy
        self.alpha = alpha
        self.phibeUpdate = phibeUpdate
        self.tau=tau
        self.delta_t = 0.1215 # 0.1215
        self.beta = np.log(self.gamma)/-self.delta_t


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
            #phi_next = self.basis_function.basisfunc(next_states[i], action)
            pn_basis_list  = [self.basis_function.basisfunc(next_states[i], a) for a in range(3)]
            pn_Qs          = [np.dot(pn_basis_list[a].T, self.policy.weights) for a in range(3)]
            pn_pis         = self.policy.softmax(pn_Qs)
            pn_phi_pi      = pn_pis @ pn_basis_list

            loss = (phi - self.gamma * pn_phi_pi)
            phi  = np.resize(phi, [k, 1])

            loss = np.resize(loss, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])

        inv_A = np.linalg.inv(A)

        new_weight= np.dot(inv_A,b)

        return new_weight
    
    def train_parameter3 (self, sample, basis_function ):
        """ Compute Q value function of current policy
            to obtain the greedy policy
        """
        k = basis_function.numWeights

        A=np.zeros([k,k])
        b=np.zeros([k,1])        

        states, actions, rewards, next_states, _ = sample
        feature_matrix = np.zeros([len(states),k])

        # function lookups
        basisfunc = self.basis_function.basisfunc
        grad_s = self.basis_function.grad_s
        softmax = self.policy.softmax
        grad_s_logit = self.policy.grad_s_logit
        softmax_pi_grad = self.policy.softmax_pi_grad
        weights = self.policy.weights
        beta = self.beta
        delta_t = self.delta_t
        np.fill_diagonal(A, 0.1)

        i = 0
        for s, a_taken, r, s_next in zip(states, actions, rewards, next_states):

            basis_list  = [basisfunc(s, a) for a in range(3)]
            phi         = basis_list[a_taken]
            Qs          = [np.dot(basis_list[a].T, weights) for a in range(3)]
            pis         = softmax(Qs)
            phi_pi      = pis @ basis_list
            grad_s_list = [grad_s(s, a) for a in range(3)]

            feature_matrix[i,:] = phi
            i += 1

            grad_logit_mat = np.array([grad_s_logit(a, grad_s_list) for a in range(3)])
            summation_b = pis @ grad_logit_mat
            #summation_b = np.sum([pis[a]*grad_s_logit(s, a) for a in range(3)], axis=0)

            grad_s_phi_pi = np.sum(
                [pis[a]*grad_s_list[a] + 
                 np.outer(basis_list[a], softmax_pi_grad(a, pis, summation_b, grad_logit_mat).T) for a in range(3)],
                 axis=0
            )
          
            
            v = s_next - s # s'-s
        
            A_i = phi - np.dot(grad_s_phi_pi, v/delta_t) - (1-beta)*phi_pi

            b_i = r 
            

            phi  = np.reshape(phi, [k, 1])

            A_real = np.matmul(phi, A_i.reshape([1,k]))
            b_real = b_i * phi

            A = A + A_real 
            b = b + b_real 
     
        new_weight= np.linalg.solve(A,b) 
        
        return new_weight
    
    def train_parameter(self, sample, basis_function):
        if self.phibeUpdate:
            return self.train_parameter3(sample, basis_function)
        else:
            return self.train_parameter1(sample, basis_function)