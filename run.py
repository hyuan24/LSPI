from matplotlib import pyplot as pl

from replay_memory import Memory
#from gym.monitoring.video_recorder import VideoRecorder
from policy import Policy
from lspi import LSPI, LSTDQ
import gymnasium as gym
import numpy as np

from env2 import ModifiedCartPoleEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

LSPI_ITERATION= 20

def test_policy(env, agent, testEps, maxSteps=3000):
   
    all_steps = []

    for j in range(testEps):
        state = env.reset(j)
  
        steps = 0
        done = False
        while not done and steps < maxSteps:
            steps += 1
         
            action=agent._act(state)
            next_state, reward, done, info, truncated = env.step(action)
            state = next_state
            
        all_steps.append(steps)
        
    final_policy = agent.policy

    return np.mean(all_steps), final_policy


def collect_data(env, memory, numEps, numPol):
    all_steps = []
    for _ in range(int(numEps*numPol)):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            steps += 1
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
          
        #print("END OF EPISODE ",j+1, "- STEPS:" ,steps)
        all_steps.append(steps)

    print(f"DATA COLLECTION COMPLETED. {memory.size()/numPol} * {numPol} samples collected. Avg episode length: {np.mean(all_steps)}")
    env.close()
    return memory, np.mean(all_steps)

def collect_uniform_data(env, memory, numEps, numPol):
    start_states = []
    next_states = []
    for _ in range(int(numEps*7*numPol)):
        
        state = env.reset()
        init_state = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        state = env.set_state([init_state])
        start_states.append(init_state)
       
      
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        memory.add([state, action, reward, next_state, done])
        next_states.append(next_state)
          
    print(f"*UNIFORM* DATA COLLECTION COMPLETED. {memory.size()/numPol} * {numPol} samples collected.")
    env.close()
    return memory, 8


def training_loop(env, memory, numPol, numEps, avg_random_steps, testEps, basisType, alpha, fancyBasis, phibeUpdate):
  
    # memory object should have samples in it!!

    test_steps = []

    for _ in tqdm(range(numPol)):
        agent = LSPI(env, env.observation_space.shape[0], basisType, alpha, 0.95, fancyBasis, phibeUpdate=phibeUpdate)
        sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
        _ = agent.train(sample, LSPI_ITERATION)
        steps, _ = test_policy(env, agent, testEps)
        test_steps.append(steps)

    return np.mean(test_steps)


def experiment_2(numPol, epRange, testEps, basisType="radial", reward="sutton_barto", alpha=1.0, uniform=False, fancyBasis=False, phibeUpdate=False):
    '''
    epRange: range of episode sizes (roughly 7 data points per episode)
    numPol: number of policies (with different samples) to train for each sample size
    testEps: number of episodes to test each policy for
    '''
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2
    totalAvg = []
    minEps, maxEps = epRange

    memory = Memory(maxEps * 10 * numPol, action_dim, obs_dim)  
    if uniform:
        memory, avg_random_steps = collect_uniform_data(env, memory, maxEps, numPol)
    else:
        memory, avg_random_steps = collect_data(env, memory, maxEps, numPol)

    for epSize in np.linspace(minEps, maxEps, int((maxEps-minEps)/100)+1):
        print(f"Sample size: {epSize} episodes.")
        
        steps = training_loop(env, memory, numPol, int(epSize), avg_random_steps, testEps, basisType, alpha, fancyBasis, phibeUpdate)
        print(f"Avg steps {steps}.")
        totalAvg.append(steps)

    return totalAvg

def plot_actions(numEps, numTicks, numPol, basisType="radial", reward="sutton_barto", alpha=1.0):
   
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2
    num_basis = 10 # PER BLOCK

    memory = Memory(numEps * 10*numPol, action_dim, obs_dim)  
    memory, avg_random_steps = collect_uniform_data(env, memory, numEps, numPol)

    actions = [0,1,2]
    angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
    angles_dot = np.linspace(-5, 5, numTicks)

    As = []
   
    for _ in range(numPol):
        agent = LSPI(env.action_space.n, num_basis, env, env.observation_space.shape[0], basisType, alpha)
        sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
        _ = agent.train(sample, LSPI_ITERATION)
        weights = agent.policy.weights

        A = np.zeros([numTicks,numTicks])

        for i, a in enumerate(angles):
            for j, ad in enumerate(angles_dot):
                state = [a,ad] 
                qs = [np.dot(agent.policy.basis_function.basisfunc(state, ac), weights) for ac in actions]
                A[i,j] = np.argmax(qs)
        As.append(A)

    fig, axs = plt.subplots(1,numPol,figsize=(4*numPol,4))
    
    for k, (ax,A) in enumerate(zip(axs,As)):
        im = ax.imshow(A, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
        if k ==0:
            ax.set_ylabel("angular velocity (rad/s)")
        ax.set_xlabel("angle (rad)")
        ax.set_title(f"policy {k+1}")
    
    fig.colorbar(im, ax=axs.ravel().tolist(), label="pi(angle, angular velocity)")
    #plt.tight_layout()
    plt.show()
        

def main():

    test1 = experiment_2(10, [400,400], 100, "radial", "dense", alpha=1.0, uniform=False, fancyBasis=True, phibeUpdate=True) 
    
    #plot_actions(400, 100, 3)
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




"""
SAMPLE SIZE 200.0 EPISODES. Avg steps 1822.42.
SAMPLE SIZE 300.0 EPISODES. sAvg steps 2611.5620000000004.
"""