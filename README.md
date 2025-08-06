# Least-Squares Policy Iteration

Implementation of LSPI (Lagoudakis and Parr) from (https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf). LSPI is an approximate model-free policy iteration method. In this case, it uses a linear architecture with radial basis functions to approximate the Q-values. 

LSTDQ (in lspi.py) iteratively updates the weights of the Q-value approximation using least-squares fixed-point approximation. LSPI (lspi.py) is essentially a main training loop that repeatedly calls LSTDQ for policy evaluation. 

Some elements of this implementation are from yusme/LSPI (https://github.com/yusme/LSPI). The CartPole environment (environment.py) is from (http://incompleteideas.net/sutton/book/code/pole.c), which is used in the Gymnasium CartPole-v1 implementation. The environment here is modified to have 3 discrete actions and noisy force, consistent with Lagoudakis and Parr. 

Results (Gymnasium parameters):
Episodes trained | Batch size | Average steps
---------------------------------------------
6000             | 6000       | 253.2284



