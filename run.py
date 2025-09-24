from matplotlib import pyplot as pl

from replay_memory import Memory
#from gym.monitoring.video_recorder import VideoRecorder
from policy import Policy
from lspi import LSPI, LSTDQ
import gymnasium as gym
import numpy as np

from environment import CartPole3 # 3 actions: left, do nothing, right
from env2 import ModifiedCartPoleEnv
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

MAX_STEPS=3000
#EPISODE = 1000 # number of training episodes
#MEMORY_SIZE= EPISODE * 24
#BATCH_SIZE = int(EPISODE) * 6 # number of samples used in training
#TEST_EPS = 100 # LSPI paper did 1000 * 100 episodes
#SEED = 0

LSPI_ITERATION= 20
#gamma = 0.95

def test_policy(env, agent, testEps):
   
    #print ("Test")
    all_steps = []

    for j in range(100):
        #if (j + 1) % 1000 == 0:
            #print(f"Test Episode #{j+1}")
    
        state = env.reset(j)
  
        steps = 0
        done = False
        while not done and steps < MAX_STEPS:
            steps += 1
         
            action=agent._act(state)
            next_state, reward, done, info, truncated = env.step(action)
            state = next_state
            
        all_steps.append(steps)
        
    final_policy = agent.policy

    return np.mean(all_steps), final_policy

def training_loop(env, memory, numPol, numEps, numBasis, testEps, basis_type):
    all_steps = []
    theta = []
    thetadot = []
    for j in range(int(500)):

        state = env.reset(1234)
        done = False
        steps = 0

        while not done:
            theta.append(state[0]) # track changes in angle and angular velocity for comparison w/ gymnasium CartPole
            thetadot.append(state[1])
            steps += 1
            action = env.action_space.sample()
            next_state, reward, done, info, truncated = env.step(action)
            #if not done: # CHANGED!!
            memory.add([state, action, reward, next_state, done])
            state = next_state
          
        #print("END OF EPISODE ",j+1, "- STEPS:" ,steps)
        all_steps.append(steps)

    print(f"DATA COLLECTION COMPLETED. {memory.containerSize} samples collected. Avg episode length: {np.mean(all_steps)}")
    env.close()
    test_steps = []
    for i in tqdm(range(25)): #env.action_space.n
        agent = LSPI(3, 10, env, 2)
        sample = memory.select_sample(int(3000))  # [current_state, actions, rewards, next_state, done]
        #print(f"Training on sample of size {sample[0].shape}")
        _ = agent.train(sample, LSPI_ITERATION)
        
        steps, policy = test_policy(env, agent, 100)
        test_steps.append(steps)

    return np.mean(test_steps)

def experiment_2(numPol, maxEps, testEps, basis_type):
    '''
    numPol: number of policies (with different samples) to train for each sample size
    numEps: maximum number of episodes to train for
    testEps: number of episodes to test each policy for
    '''
    env = ModifiedCartPoleEnv()
    action_dim = 1
    obs_dim = 2
    num_basis = 10 # PER BLOCK
    totalAvg = []

    for epSize in np.linspace(500, maxEps, 1):
        print(f"Memory size: {epSize*9}.")
        memory = Memory(4500, 1, 2)
        
       
        steps = training_loop(env, memory, 25, 500, 10, 100, "radial")
        #f training_loop(env, memory, numPol, numEps, numBasis, testEps, basis_type):
        print(f"Avg steps {steps}.")
        totalAvg.append(steps)

    return totalAvg


def main():

    #print(experiment_2(25, 1000, 50))
    experiment_2(25, 500, 100, "radial")
    '''
    results = np.array([25.33, 2498.54, 2653.33, 3000, 3000, 3000, 3000, 3000, 3000, 3000]) 

    plt.figure()
    plt.plot(np.linspace(100, 1000,10), results, label='Average episode length')
    plt.xlabel("Training Episodes")
    plt.ylabel("Average steps per episode")
    plt.legend()
    plt.grid(True)
    plt.show() 
    '''
         
# 17.4334, 17.4420, 11.9434 (radial), 
# 6.91088, 8.00192, 7.0058 (poly)


if __name__ == '__main__':
    main()





