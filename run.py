from matplotlib import pyplot as pl

from replay_memory import Memory
#from gym.monitoring.video_recorder import VideoRecorder
from policy import Policy
from lspi import LSPI, LSTDQ
import gymnasium as gym
import numpy as np

from environment import CartPole3 # 3 actions: left, do nothing, right
import matplotlib.pyplot as plt
from collections import defaultdict

MAX_STEPS=15000
EPISODE = 1000 # number of training episodes
MEMORY_SIZE= EPISODE * 24
BATCH_SIZE = int(EPISODE) * 6 # number of samples used in training
TEST_EPS = 5000 # LSPI paper did 1000 * 100 episodes
SEED = 0

lspi_iteration = 100
gamma = 0.95

def test_policy(env, agent):
   
    print ("Test")
    all_steps = []

    for j in range(TEST_EPS):
        if (j + 1) % 1000 == 0:
            print(f"Test Episode #{j+1}")
        state = env.reset()
  
        steps = 0
        done = False
        while not done and steps < MAX_STEPS:
            steps += 1
         
            action=agent._act(state)
            next_state, reward, done, info, truncated = env.step(action)
            state = next_state
            
        all_steps.append(steps)
        
    final_policy = agent.policy

    return all_steps, final_policy

def training_loop(env, memory, agent):
    all_steps = []
    theta = []
    thetadot = []
    for j in range(EPISODE):
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
        
    '''
    plt.figure()
    plt.plot(range(len(theta)), theta, label='Theta')
    plt.plot(range(len(thetadot)), thetadot, label='Thetadot')

    plt.grid(True)
    plt.show()
    '''

    print(f"DATA COLLECTION COMPLETED. AVG EPISODE LENGTH: {np.mean(all_steps)}")
    env.close()

    sample = memory.select_sample(BATCH_SIZE)  # [current_state, actions, rewards, next_state, done]
    _ = agent.train(sample, lspi_iteration)
    steps, policy = test_policy(env, agent)

    return steps, policy

def experiment_1():
    #env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v1')
    #env = NoisyDiscretePendulum()
    #env = ModifiedCartPoleEnv()
    _ = env.reset()

    action_dim = 1
    obs_dim = 2
    num_actions = env.action_space.n
    num_basis = 10 # PER BLOCK
    
    memory = Memory(MEMORY_SIZE, action_dim,  obs_dim)

    agent = LSPI(num_actions, num_basis, env, obs_dim)

    return agent, env, memory

def experiment_2():
    env = CartPole3()
    _ = env.reset()

    action_dim = 1
    obs_dim = 2
    num_actions = env.action_space.n
    num_basis = 10 # PER BLOCK
    
    memory = Memory(MEMORY_SIZE, action_dim,  obs_dim)

    agent = LSPI(num_actions, num_basis, env, obs_dim)

    return agent, env, memory


def main():

    agent, env, memory = experiment_2()

    #print("memory size", memory.containerSize)

    steps, _ = training_loop(env, memory, agent)
    print(np.mean(steps))
   


if __name__ == '__main__':
    main()





