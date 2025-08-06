from scipy import *
from scipy.linalg import norm, pinv
import itertools

import numpy as np
import math

class BasisFunction:
    def __init__(self, indim,  outdim, numWeights, param="gym"):
        self.indim = indim
        self.outdim = outdim
        self.numWeights = numWeights

        if param=="gym":
            theta, thetadot = [[-0.1, 0, 0.1], [-1, 0, 1]]
        elif param=="lagoudakis":
            theta, thetadot = [[-math.pi/4, 0, math.pi/4], [-1, 0, 1]]

        self.e_centers = np.array(list(itertools.product(theta, thetadot))) # non-constant centers
        print(f"e_centers DIM: {np.shape(self.e_centers)}")

        self.beta = 1/2

    def basisfunc(self, state, action):
        assert len(state) == self.indim
      
        #norm_1 = (c-d)/((c**2)+(d**2))**(1/2)
        basis = []
        block = np.zeros(10)
        block[0] = 1  # constant term
        for i, c in enumerate(self.e_centers):
            block[i+1] = np.exp(-self.beta * np.linalg.norm(c - state)**2)

        #print(block)
        
        if action == 0: # THREE ACTIONS
            basis = np.append(block, np.zeros(20))
        elif action == 1:
            basis = np.append(np.append(np.zeros(10), block), np.zeros(10))
        elif action == 2:
            basis = np.append(np.zeros(20), block) # CHANGED
        else:
            ValueError("Invalid Action in rbf.py basisfunc")
        '''
        if action == 0: # TWO ACTIONS
            basis = np.append(block, np.zeros(10))
        elif action == 1:
            basis = np.append(np.zeros(10), block)
        else:
            ValueError("Invalid Action in rbf.py basisfunc")
        '''
        return basis

if __name__ == '__main__':
    print("Hello")
    rbf = BasisFunction(2, 1, 20)
    print(rbf.basisfunc([-np.pi/4, -1], 1))
