from scipy import *
import itertools

import numpy as np
import math

class RadialBasisFunction:
    def __init__(self, indim,  outdim, actionBasis=False):
        self.indim = indim
        self.outdim = outdim
        self.actionBasis = actionBasis

        theta, thetadot = [[-math.pi/4, 0, math.pi/4], [-1, 0, 1]]

        self.e_centers = np.array(list(itertools.product(theta, thetadot))) # non-constant centers
        self.action_centers = np.array([0, 1, 2])
        self.numWeights=(len(self.e_centers)+1)*(len(self.action_centers)+1)

        self.beta = 1/2 
        self.action_beta = 1/2 
        self.alpha = self.beta 
        self.theta_scale = 1
        self.theta_dot_scale = 1
    
    def basisfunc1(self, state, action): # ORIGINAL BASIS
        assert len(state) == self.indim
      
        state = np.asarray([state[0]/self.theta_scale, state[1]/self.theta_dot_scale])

        basis = []
        block = np.zeros(10)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            c = np.asarray([c[0]/self.theta_scale, c[1]/self.theta_dot_scale])
            block[i+1] = np.exp(-self.alpha * np.linalg.norm(c - state)**2)
        
        if action == 0:
            basis = np.concatenate([block, np.zeros(20)])
        elif action == 1:
            basis = np.concatenate([np.zeros(10), block, np.zeros(10)])
        elif action == 2:
            basis = np.concatenate([np.zeros(20), block])
        else:
            raise ValueError("Invalid Action in rbf.py basisfunc")
      
        return basis
    
    def basisfunc_scaled(self, state, action):
  
        #state = [state[0]/self.theta_scale, state[1]/self.theta_dot_scale]

        assert len(state) == self.indim
      
        block = np.zeros(len(self.e_centers)+1)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            #c = np.asarray([c[0]/self.theta_scale, c[1]/self.theta_dot_scale])
            diff = state - c
            sq = np.sum(diff * diff)
            block[i+1] = np.exp(-self.alpha * sq)
    
        action_block = np.zeros(len(self.action_centers)+1)
        action_block[0] = 1
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = np.exp(-self.action_beta * (action - c)**2)
     
        prod_grid = block[:, None] * action_block[None, :]
        basis = prod_grid.ravel()

        return basis
    
    def basisfunc2(self, state, action): # MODIFIED BASIS W ACTION CENTERS
        return self.basisfunc_scaled(state, action)
        assert len(state) == self.indim
      
        block = np.zeros(len(self.e_centers)+1)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            block[i+1] = np.exp(-self.beta * np.linalg.norm(state - c)**2)
    
        action_block = np.zeros(len(self.action_centers) +1)
        action_block[0] = 1
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = np.exp(-self.action_beta * (action - c)**2)
     
        prod_grid = block[:, None] * action_block[None, :]
        basis = prod_grid.ravel()

        return basis
    
 
    
    def basisfunc(self, state, action):
        if self.actionBasis:
            return self.basisfunc2(state, action)
        else:
            return self.basisfunc1(state, action)

    def grad_s(self, state, action):
        if self.actionBasis:
            return self.grad_s_scaled(state, action)
        else:
            assert len(state) == self.indim

            grad = np.zeros([30, self.indim])
            block = np.zeros([10, self.indim])
            state = np.asarray([state[0]/self.theta_scale, state[1]/self.theta_dot_scale], dtype=np.float64)
            for i, c in enumerate(self.e_centers):
                c = np.asarray([c[0]/self.theta_scale, c[1]/self.theta_dot_scale])
                diff = c - state
                rbf = np.exp(-self.alpha * np.linalg.norm(diff)**2)
                block[i+1,0] = 2*self.alpha* diff[0]* rbf/ self.theta_scale
                block[i+1,1] = 2*self.alpha* diff[1]* rbf/ self.theta_dot_scale
            
            grad[(action*10):(action*10+10), ] = block
      
            return grad
    
        block = np.zeros([len(self.e_centers)+1,self.indim])
        for i, c in enumerate(self.e_centers):
            block[i+1,] = (c - state) * np.exp(-self.beta * np.linalg.norm(state - c)**2)
        action_block = np.zeros([len(self.action_centers)+1,1])
        action_block[0]=1
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = np.exp(-self.action_beta * (action - c)**2)
        prod_grid = block[:, None] * action_block[None, :]
        grad = prod_grid.reshape(-1,2)
        return grad
    
    def grad_s_scaled(self, state, action):
    

        #s = np.asarray([state[0] / self.theta_scale, state[1] / self.theta_dot_scale], dtype=np.float64)
        s = state

        block = np.zeros((len(self.e_centers)+1, self.indim))

        for i, c in enumerate(self.e_centers):
            #c = np.asarray([c[0] / self.theta_scale, c[1] / self.theta_dot_scale], dtype=np.float64)
            diff = s - c
            sq = np.sum(diff * diff)
            rbf = np.exp(-self.alpha * sq)

            block[i+1, 0] = -2 * self.alpha * diff[0] * rbf / self.theta_scale
            block[i+1, 1] = -2 * self.alpha * diff[1] * rbf / self.theta_dot_scale

        action_block = np.zeros((len(self.action_centers)+1, 1))
        action_block[0] = 1.0
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = np.exp(-self.action_beta * (action - c) ** 2)

        prod_grid = block[:, None] * action_block[None, :]
        grad = prod_grid.reshape(-1, 2)
        return grad

    def grad_a(self, state, action):

        block = np.zeros([len(self.e_centers)+1,1])
        block[0] = 1
        for i, c in enumerate(self.e_centers):
            block[i+1] = np.exp(-self.beta * np.linalg.norm(c - state)**2)

        action_block = np.zeros([len(self.action_centers)+1,1])
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = (c-action) * np.exp(-self.action_beta * (action - c)**2)

        prod_grid = block[:, None] * action_block[None, :]
        grad = prod_grid.ravel()

        return grad

    ##-----------------------------##
    #          NEW STUFF           ##
    ##-----------------------------##

    def hess_ss(self, state, action):
 
        nE = len(self.e_centers)
        nA = len(self.action_centers)

        Hs_block = np.zeros((nE + 1, self.indim, self.indim))
        I = np.eye(self.indim)

        for i, c in enumerate(self.e_centers):
            v = (c - state)  
            g = np.exp(-self.beta * np.linalg.norm(state - c) ** 2)
            Hs_block[i + 1] = (np.outer(v, v) - I) * g

        a_block = np.zeros((nA + 1,))
        a_block[0] = 1.0
        for j, c in enumerate(self.action_centers):
            a_block[j + 1] = np.exp(-self.action_beta * (action - c) ** 2)

        prod = Hs_block[:, None, :, :] * a_block[None, :, None, None]
        return prod.reshape(-1, self.indim, self.indim)


    def hess_aa(self, state, action):

        nE = len(self.e_centers)
        nA = len(self.action_centers)

        s_block = np.zeros((nE + 1, 1))
        s_block[0] = 1.0
        for i, c in enumerate(self.e_centers):
            s_block[i + 1] = np.exp(-self.beta * np.linalg.norm(c - state) ** 2)

        Haa_block = np.zeros((nA + 1, 1))  
        for j, c in enumerate(self.action_centers):
            v = (c - action) 
            h = np.exp(-self.action_beta * (action - c) ** 2)
            Haa_block[j + 1] = (v * v - 1.0) * h

        prod = s_block[:, None] * Haa_block[None, :]
        return prod.ravel()


    def hess_sa(self, state, action):

        nE = len(self.e_centers)
        nA = len(self.action_centers)

        gs_block = np.zeros((nE + 1, self.indim))
        for i, c in enumerate(self.e_centers):
            gs_block[i + 1] = (c - state) * np.exp(-self.beta * np.linalg.norm(state - c) ** 2)

        ga_block = np.zeros((nA + 1,)) 
        for j, c in enumerate(self.action_centers):
            ga_block[j + 1] = (c - action) * np.exp(-self.action_beta * (action - c) ** 2)

        prod = gs_block[:, None, :] * ga_block[None, :, None]
        return prod.reshape(-1, self.indim)


    def hess_as(self, state, action):

        return self.hess_sa(state, action)


if __name__ == '__main__':
    rbf = RadialBasisFunction(2, 1, True)
    s =[np.pi/3, 1.1]
    a = 2
    print(rbf.basisfunc_scaled(s, a))
    s = [np.pi/4, -1]
    a = 0
    print(rbf.basisfunc_scaled(s, a))
    

