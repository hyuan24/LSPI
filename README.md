# Least-Squares Policy Iteration

Implementation of LSPI (Lagoudakis and Parr) from [https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf]. LSPI is an approximate model-free policy iteration method which uses a linear architecture with radial basis functions to approximate the Q-values. 

LSTDQ (in lspi.py) iteratively updates the weights of the Q-value approximation using least-squares fixed-point approximation. In particular, here we modify the update based on the PhiBE framework, which proposes a new differential equation-based Bellman equation. 

Some elements of this implementation are from [github.com/yusme/LSPI].  






