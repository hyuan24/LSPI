from scipy import *
from scipy.linalg import norm, pinv
import itertools

import numpy as np
import math

class RadialBasisFunction:
    def __init__(self, indim,  outdim):
        self.indim = indim
        self.outdim = outdim
        self.numWeights = 30

        #theta, thetadot = [[-math.pi/4, 0, math.pi/4], [-1, 0, 1]]
        theta, thetadot = [[-1, 0, 1], [-1, 0, 1]]

        self.e_centers = np.array(list(itertools.product(theta, thetadot))) # non-constant centers
        self.action_centers = np.array([0.75,1.25])

        self.beta = 1/2
        self.action_beta = 1/2
        self.action_A = 1

    def basisfunc(self, state, action):
        assert len(state) == self.indim
      
        block = np.zeros(len(self.e_centers)+1)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            block[i+1] = np.exp(-self.beta * np.linalg.norm(state - c)**2)
    
        action_block = np.zeros(len(self.action_centers) +1)
        action_block[0] = 1
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = self.action_A * np.exp(-self.action_beta * (action - c)**2)
     
        prod_grid = block[:, None] * action_block[None, :]
        basis = prod_grid.ravel()

        return basis
    
    def grad_s(self, state, action):

        block = np.zeros([len(self.e_centers)+1,self.indim])
        for i, c in enumerate(self.e_centers):
            block[i+1,] = (c - state) * np.exp(-self.beta * np.linalg.norm(state - c)**2)

        action_block = np.zeros([len(self.action_centers)+1,1])
        action_block[0]=1
        for i, c in enumerate(self.action_centers):
            action_block[i+1] = self.action_A * np.exp(-self.action_beta * (action - c)**2)

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
            action_block[i+1] = (c-action) * self.action_A * np.exp(-self.action_beta * (action - c)**2)

        prod_grid = block[:, None] * action_block[None, :]
        grad = prod_grid.ravel()

        return grad

class PolynomialBasisFunction:
    """
    - degree: total polynomial degree (includes cross terms).
      For indim=2, degree=3 -> 10 terms: 1, x1, x2, x1^2, x1 x2, x2^2, x1^3, x1^2 x2, x1 x2^2, x2^3
    """
    def __init__(self, indim, outdim, degree=3):
        #assert indim >= 1
        self.indim = indim
        #self.outdim = outdim
        self.degree = degree
 
        self.exponents = []
        for total_deg in range(degree + 1):
            for exps in itertools.product(range(total_deg + 1), repeat=indim):
                if sum(exps) == total_deg:
                    self.exponents.append(exps)
        #print(self.exponents)

        self.block_size = len(self.exponents)  
        self.numWeights = self.block_size * 3
        
    def _poly_block(self, x):
        """
        Evaluate all monomials x^exponents for exponents in self.exponents.
        Order is by total degree, then lexicographic within degree.
        """
        feats = np.empty(self.block_size, dtype=float)
        for i, exps in enumerate(self.exponents):
            # x^exps = prod_j x[j]**exps[j]
            val = 1.0
            for j, e in enumerate(exps):
                if e:
                    val *= x[j] ** e
            feats[i] = val
        return feats

    def basisfunc(self, state, action):
        """
        Returns a feature vector with three action-specific blocks:
        [ block(state) | 0 | 0 ]        for action=0
        [ 0 | block(state) | 0 ]        for action=1
        [ 0 | 0 | block(state) ]        for action=2
        """
        block = self._poly_block(state)  # shape: (block_size,)

        if action == 0:
            basis = np.concatenate([block, np.zeros(2 * self.block_size)])
        elif action == 1:
            basis = np.concatenate([np.zeros(self.block_size), block, np.zeros(self.block_size)])
        elif action == 2:
            basis = np.concatenate([np.zeros(2 * self.block_size), block])
        else:
            raise ValueError("Invalid Action in polynomial basisfunc (expected 0, 1, or 2)")

        return basis



if __name__ == '__main__':
    #print("Hello")
    rbf = RadialBasisFunction(2, 1)
    
    print(rbf.basisfunc([0.5, -3], 2))
    print(rbf.grad_s([0.5, -3], 2))
