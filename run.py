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
from rbf import RadialBasisFunction

LSPI_ITERATION = 40

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


def collect_data(env, memory, maxEps, numPol):
    all_steps = []
    for _ in range(int(maxEps*numPol*7)):
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

    print(f"DATA COLLECTION COMPLETED. {memory.size()/numPol} * {numPol} samples collected. Avg episode lengthH: {np.mean(all_steps)}")
    env.close()
    return memory, np.mean(all_steps)

"""
def collect_data_uniform(env, memory, maxEps, numPol):
    for _ in range(int(maxEps*numPol)*7*7):
        state = env.reset()
        state = np.random.uniform([-np.pi/2, -6], [np.pi/2, 6])
        env.set_state(state)

        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        memory.add([state, action, reward, next_state, done])
              

    print(f"*UNIFORM* DATA COLLECTION COMPLETED. {memory.size()/numPol} * {numPol} samples collected.")
    env.close()
    return memory, 7
"""


def training_loop(env, memory, numPol, numEps, avg_random_steps, testEps, basisType, alpha):
  
    # memory object should have samples in it!!

    test_steps = []

    for _ in tqdm(range(numPol)):
        agent = LSPI(env, env.observation_space.shape[0], basisType, alpha)
        sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
        policy = agent.train(sample, LSPI_ITERATION)
        steps, _ = test_policy(env, agent, testEps)
        test_steps.append(steps)

    return np.mean(test_steps), policy.weights


def experiment_2(numPol, maxEps, testEps, basisType="radial", reward="dense", alpha=1.0):
    '''
    numPol: number of policies (with different samples) to train for each sample size
    numEps: maximum number of episodes to train for
    testEps: number of episodes to test each policy for
    '''
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2
    totalAvg = []

    memory = Memory(maxEps * 10 * numPol * 7, action_dim, obs_dim)  
    memory, avg_random_steps = collect_data(env, memory, maxEps, numPol)

    for epSize in np.linspace(300, maxEps, 1):
        print(f"Sample size: {epSize} episodes. Avg steps per episode: {avg_random_steps}.")
        
        steps, weights = training_loop(env, memory, numPol, epSize, avg_random_steps, testEps, basisType, alpha)
        print(f"Avg steps {steps}.")
        totalAvg.append(steps)

    return totalAvg, weights

def plot_actions(numEps, numTicks, numPol, basisType="radial", reward="dense", alpha=1.0):
   
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2

    memory = Memory(numEps * 10 * 7, action_dim, obs_dim)  
    memory, avg_random_steps = collect_data(env, memory, numEps, 1)

    angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
    angles_dot = np.linspace(-6, 6, numTicks)
    actions = [0,1,2]
    matrices = []

    for _ in range(numPol):
        agent = LSPI(env, obs_dim, basisType, alpha)
        sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
        _ = agent.train(sample, LSPI_ITERATION)
        weights = agent.policy.weights

        A = np.zeros([numTicks,numTicks])

        for i, a in enumerate(angles):
            for j, ad in enumerate(angles_dot):
                state = [a,ad] 
                qs = [np.dot(agent.policy.basis_function.basisfunc(state, ac), weights) for ac in actions]
                A[i,j] = np.argmax(qs)
        matrices.append(A)

    fig, axs = plt.subplots(1,3,figsize=(12,4))
    for k, (ax,A) in enumerate(zip(axs,matrices)):
        im = ax.imshow(A, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
        
        ax.set_xlabel("angle (rad)")
        if k ==0:
            ax.set_ylabel("angular velocity (rad/s)")
        ax.set_title(f"policy {k+1}")
    fig.colorbar(im, ax=axs.ravel().tolist(), label="pi(angle, angular velocity)")
    #plt.tight_layout()
    plt.show()

def main():

    #test1, weights = experiment_2(10, 300, 100, "radial", "dense") 
    #plot_values(rbf, weights)
    plot_actions(300, 100, 3, "radial", "dense", 1.0)
    #data = np.column_stack((poly,poly))
    #np.savetxt("results.txt", data, header="poly poly", comments='')
    

if __name__ == '__main__':
    main()


"""
SAMPLE SIZE 200.0 EPISODES. Avg steps 1822.42.
SAMPLE SIZE 300.0 EPISODES. Avg steps 2611.5620000000004.
"""