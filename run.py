from matplotlib import pyplot as pl

from replay_memory import Memory
from lspi import LSPI
import numpy as np

from env2 import ModifiedCartPoleEnv
import matplotlib.pyplot as plt
from tqdm import tqdm

LSPI_ITERATION= 20
GAMMA=0.975

def test_policy(env, agent, testEps, maxSteps=50000):
    #plot_actions(0,200,3, "none", "none", 1, False, False, True, agent)
    all_steps = []
    perstep_rewards = []
    for j in range(testEps):
        cum_reward = 0
        state = env.reset(j)
  
        steps = 0
        done = False
        while not done and steps < maxSteps:
           
            steps += 1
         
            action=agent._act(state)
            next_state, reward, done, info, truncated = env.step(action)
            cum_reward += reward
            state = next_state
            
        all_steps.append(steps)
        perstep_rewards.append(cum_reward/steps)

    #final_policy = agent.policy
    print(f"Policy ran {np.mean(all_steps)} steps and accumulated {np.mean(perstep_rewards)} per step")
    return np.mean(all_steps), np.mean(perstep_rewards)


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

            #memory.add([-1*state, 2-action, reward, -1*next_state, done])
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


def training_loop(env, testEnv, memory, numPol, numEps, avg_random_steps, testEps, basisType, alpha, fancyBasis, phibeUpdate, tau):
  
    # memory object should have samples in it!!

    test_steps = []
    test_rewards = []
    #test_rewards = []

    for _ in tqdm(range(numPol)):
        agent = LSPI(env, env.observation_space.shape[0], basisType, alpha, GAMMA, tau, fancyBasis, phibeUpdate=phibeUpdate)
        sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
        _ = agent.train(sample, LSPI_ITERATION)
        steps, reward = test_policy(testEnv, agent, testEps)
        test_steps.append(steps)
        test_rewards.append(reward)
        

    return np.mean(test_steps), np.mean(test_rewards)


def experiment_2(numPol, epRange, testEps, tau, basisType="radial", reward="sutton_barto", alpha=1.0, uniform=False, fancyBasis=False, phibeUpdate=False, testTau=0.1215):
    '''
    epRange: range of episode sizes (roughly 7 data points per episode)
    numPol: number of policies (with different samples) to train for each sample size
    testEps: number of episodes to test each policy for
    '''
    env = ModifiedCartPoleEnv(reward)
    testEnv = ModifiedCartPoleEnv(reward, testTau)
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
        
        steps, rew = training_loop(env, testEnv, memory, numPol, int(epSize), avg_random_steps, testEps, basisType, alpha, fancyBasis, phibeUpdate, tau)
        print(f"Avg steps {steps}. Avg reward {rew} per step")
        totalAvg.append(steps)

    return totalAvg

def plot_actions(numEps, numTicks, numPol, tau, basisType="radial", reward="sutton_barto", alpha=1.0, uniform=False, fancyBasis=False, phibeUpdate=False, useAgent=None, soft=False):
   
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2

    angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
    angles_dot = np.linspace(-5, 5, numTicks)
    As = []

    if useAgent is None:
        memory = Memory(numEps * 10*numPol, action_dim, obs_dim)  
        if uniform:
            memory, avg_random_steps = collect_uniform_data(env, memory, numEps, numPol)
        else: 
            memory, avg_random_steps = collect_data(env, memory, numEps, numPol)
   
    for _ in tqdm(range(numPol)):
        if useAgent is None:
            agent = LSPI(env, env.observation_space.shape[0], basisType, alpha, gamma=GAMMA, tau=tau, fancyBasis=fancyBasis, phibeUpdate=phibeUpdate)
            sample = memory.select_sample(round(numEps*avg_random_steps))  # [current_state, actions, rewards, next_state, done]
            _ = agent.train(sample, LSPI_ITERATION)
        else:
            agent = useAgent

        A = np.zeros([numTicks,numTicks])

        for i, a in enumerate(angles):
            for j, ad in enumerate(angles_dot):
                state = [a,ad] 
                if soft:
                    A[i,j] = agent.policy.softmax_get_action(state)
                else:
                    A[i,j] = agent.policy.get_actions(state)
        As.append(A)

    # GRAPH STUFF
    fig, axs = plt.subplots(1,numPol,figsize=(4*numPol,4))
    
    for k, (ax,A) in enumerate(zip(axs,As)):
        im = ax.imshow(A, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
        if k ==0:
            ax.set_ylabel("angular velocity (rad/s)")
        ax.set_xlabel("angle (rad)")
        ax.set_title(f"Policy {k+1}")
    
    fig.colorbar(im, ax=axs.ravel().tolist(), label="pi(state)")
    plt.show()
        
def plot_qs(numEps, numTicks, tau, basisType="radial", reward="sutton_barto", alpha=1.0, uniform=False, fancyBasis=False, phibeUpdate=False, type="learned_qs"):
   
    env = ModifiedCartPoleEnv(reward)
    action_dim = 1
    obs_dim = 2
    delta_t = 1

    memory = Memory(numEps * 12, action_dim, obs_dim)  
    if uniform:
        memory, avg_random_steps = collect_uniform_data(env, memory, numEps, 1)
    else: 
        memory, avg_random_steps = collect_data(env, memory, numEps, 1)

    actions = [0,1,2]
    angles = np.linspace(-np.pi/2, np.pi/2, numTicks)
    angles_dot = np.linspace(-5, 5, numTicks)

    Qs = []
    sample = memory.select_sample(memory.size()) 


    if not phibeUpdate or type=="phibe_estimate":
        agent = LSPI(env, env.observation_space.shape[0], basisType, alpha, gamma=GAMMA, tau=tau, fancyBasis=fancyBasis)
        _ = agent.train(sample, LSPI_ITERATION)
        weights = agent.policy.weights
    else:
        agent = LSPI(env, env.observation_space.shape[0], basisType, alpha, gamma=GAMMA, tau=tau, fancyBasis=fancyBasis, phibeUpdate=phibeUpdate)
        _ = agent.train(sample, LSPI_ITERATION)
        weights = agent.policy.weights
    print("Finished training.")

    for k in range(len(actions)):

        Q = np.zeros([numTicks,numTicks])

        for i, a in enumerate(angles):
            for j, ad in enumerate(angles_dot):
                state = [a,ad] 
                #action = agent.policy.get_actions(state)
                if type=="learned_qs": # ACTUALLY V VALUE NOW!
                    pair = np.mod(k+1,3)
                    q1 = np.dot(agent.policy.basis_function.basisfunc(state, k), weights)
                    q2 = np.dot(agent.policy.basis_function.basisfunc(state, pair), weights)
                    q = q1-q2
                elif type=="phibe_estimate":
                    Q_rl = np.dot(agent.policy.basis_function.basisfunc(state, k), weights)
                    V_rl = np.dot(agent.policy.basis_function.basisfunc(state, agent.policy.get_actions(state)), weights)
                    q = (Q_rl-V_rl)/delta_t + V_rl
                Q[i,j] =q
        Qs.append(Q)

    fig, axs = plt.subplots(1,3,figsize=(4,4))
    
    for k, (ax,Q) in enumerate(zip(axs,Qs)):
        im = ax.imshow(Q, origin="lower",extent=[angles[0], angles[-1], angles_dot[0], angles_dot[-1]], aspect='auto')
        if k ==0:
            ax.set_ylabel("angular velocity (rad/s)")
        ax.set_xlabel("angle (rad)")
        ax.set_title(f"Q(state, action={k}-{np.mod(k+1,3)})")
    
    fig.colorbar(im, ax=axs.ravel().tolist(), label="Q")
    plt.show()

def main():
    import matplotlib.pyplot as plt
    _ = experiment_2(30, [1000,1000], 45, 100, "radial", "dense", alpha=1, uniform=False, fancyBasis=True, phibeUpdate=False, testTau=0.01215/10) 

    _ = experiment_2(30, [1000,1000], 45, 100, "radial", "dense", alpha=1, uniform=False, fancyBasis=True, phibeUpdate=True, testTau=0.01215/10) 

    #_ = experiment_2(30, [1000,1000], 60, 1, "radial", "dense", alpha=1, uniform=False, fancyBasis=True, phibeUpdate=False, testTau=0.01215) 
    
    #plot_actions(1000, 150, 2, 4, basisType="radial", reward="dense", alpha=1, uniform=False, fancyBasis=True, phibeUpdate=True, soft=True)
 
    #plot_qs(1000,100, 1.75, "radial", "dense", 1, False, False, True, "learned_qs")

    # Avg steps 50000.0. Avg reward -0.007355221520073872 per step PhiBE
    # Avg steps 40094.96. Avg reward -0.06692162762982254 per step LSPI

if __name__ == '__main__':
    main()