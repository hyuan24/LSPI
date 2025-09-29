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
EPISODE = 1000 # number of training episodes
MEMORY_SIZE= EPISODE * 24
BATCH_SIZE = int(EPISODE) * 6 # number of samples used in training
TEST_EPS = 1000 # LSPI paper did 1000 * 100 episodes
SEED = 0

LSPI_ITERATION= 20
gamma = 0.95

def test_policy(env, agent, testEps):
   
    #print ("Test")
    all_steps = []

    for j in range(testEps):
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
        print(f"Test run {j+1}: {steps} steps.")
        
    final_policy = agent.policy

    return np.mean(all_steps), final_policy

def training_loop(env, memory, numPol, numEps, numBasis, testEps, basisType, alpha):
    all_steps = []
    theta = []
    thetadot = []
    #for j in range(int(numEps*3/4*numPol)): REVERT LATER
    for j in range(int(numEps*numPol)):
        state = env.reset()
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

    print(f"DATA COLLECTION COMPLETED. {memory.size()/numPol} * {numPol} episodes collected. Avg episode lengthH: {np.mean(all_steps)}")
    env.close()
    test_steps = []
    for i in tqdm(range(numPol)):
        agent = LSPI(env.action_space.n, numBasis, env, env.observation_space.shape[0], basisType, alpha)
        sample = memory.select_sample(int(memory.size()//numPol))  # [current_state, actions, rewards, next_state, done]
        _ = agent.train(sample, LSPI_ITERATION)
        steps, policy = test_policy(env, agent, testEps)
        test_steps.append(steps)

    return np.mean(test_steps)


def experiment_2(numPol, maxEps, testEps, basisType, reward, alpha=1.0):
    '''
    numPol: number of policies (with different samples) to train for each sample size
    numEps: maximum number of episodes to train for
    testEps: number of episodes to test each policy for
    '''
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2
    num_basis = 10 # PER BLOCK
    totalAvg = []

    for epSize in np.linspace(300, maxEps, 1):
        print(f"SAMPLE SIZE {epSize} EPISODES.")
        memory = Memory(epSize * 10 * numPol, action_dim,  obs_dim)
        
       
        steps = training_loop(env, memory, numPol, epSize, num_basis, testEps, basisType, alpha)
        print(f"Avg steps {steps}.")
        totalAvg.append(steps)

    return totalAvg


def main():

    test1 = experiment_2(10, 300, 100, "radial", "sutton_barto", alpha=1.0) # 92, 766

    #data = np.column_stack((poly,poly))
    #np.savetxt("results.txt", data, header="poly poly", comments='')

    '''
    plt.figure()
    plt.plot(np.linspace(100, 1000,10), results, label='Average episode length')
    plt.xlabel("Training Episodes")
    plt.ylabel("Average steps per episode")
    plt.legend()
    plt.grid(True)
    plt.show() 
    '''
    



if __name__ == '__main__':
    main()





