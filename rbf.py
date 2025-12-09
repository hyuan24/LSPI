from scipy import *
from scipy.linalg import norm, pinv
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
        self.numWeights = (len(self.e_centers)+1)*(len(self.action_centers)+1) if self.actionBasis else 30

        self.beta = 1/2
        self.action_beta = 1/2
    
    def basisfunc1(self, state, action): # ORIGINAL BASIS
        assert len(state) == self.indim
      
        basis = []
        block = np.zeros(10)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            block[i+1] = np.exp(-self.beta * np.linalg.norm(c - state)**2)
        
        if action == 0: # THREE ACTIONS
            basis = np.append(block, np.zeros(20))
        elif action == 1:
            basis = np.append(np.append(np.zeros(10), block), np.zeros(10))
        elif action == 2:
            basis = np.append(np.zeros(20), block) 
        else:
            ValueError("Invalid Action in rbf.py basisfunc")
      
        return basis
    
    def basisfunc2(self, state, action): # MODIFIED BASIS W ACTION CENTERS
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



if __name__ == '__main__':
    rbf = RadialBasisFunction(2, 1, True)
    print(rbf.basisfunc([np.pi/4, 1], 1))
    print(rbf.grad_a([np.pi/4, 1], 2))
    

