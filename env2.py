import math
from typing import Union
import string

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from scipy.integrate import solve_ivp

class ModifiedCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, reward: string = "sutton_barto", render_mode: str | None = None
    ):
        self.reward = reward
        self.param = "lagoudakis"
        self.gravity = 9.8
        self.masscart = 8.0 # 8.0
        self.masspole = 2.0 # 2.0
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 0.25
        self.polemass_length = self.masspole * self.length
        self.force_mag = 50.0 # 50
        self.tau = 0.1215  # seconds between state updates # 0.1
        self.kinematics_integrator = "RK45"

        # Angle at which to fail the episode
        #self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_threshold_radians = math.pi / 2
        self.x_threshold = np.inf

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(3) # CHANGED
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self.render_mode = render_mode
        
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
    
        
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        prev_theta, _ = self.state
        forces = [-self.force_mag, 0, self.force_mag]
        force_noise = np.random.uniform(-10, 10)
        force = forces[action] + force_noise
    
        self.state = self.integrate(force)
        
        terminated = bool(
            prev_theta < -self.theta_threshold_radians
            or prev_theta > self.theta_threshold_radians
        )
        if self.reward=="sutton_barto":
            if not terminated:
                reward = 0.0 
            elif self.steps_beyond_terminated is None:
            # Pole just fell!
                self.steps_beyond_terminated = 0
                reward = -1.0 
            else:
                if self.steps_beyond_terminated >= 0:
                    logger.warn(
                        "You are calling 'step()' even though this environment has already returned terminated = True. "
                        "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward = -1.0 
        elif self.reward=="dense":
            reward = np.cos(prev_theta)-1
        else:
            ValueError()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        #return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
        return self.state, reward, terminated, False, {}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = -0.2, 0.2
        #np.random.seed(seed)
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        self.steps_beyond_terminated = None

        return self.state
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def set_state(self, state):
        if len(state)==1:
            self.state[0] = state[0]
        if len(state)==2:
            self.state = state
        return self.state

    def integrate(self, force):
        #state = self.state
        def cartpole_dynamics(t, y):
            theta, theta_dot = y
    
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            alpha = 1.0/self.total_mass
            theta_acc = (self.gravity * sintheta - alpha * 
                         self.masspole * self.length * theta_dot**2 * np.sin(2 * theta) 
                         / 2 - alpha * costheta * force) / ( (4.0 / 3.0) * self.length - alpha * self.masspole * self.length * costheta**2 )
            return [theta_dot, theta_acc]
        
        sol = solve_ivp(cartpole_dynamics, [0, self.tau], self.state, method="RK45", t_eval=[self.tau])
        return np.array(sol.y[:, -1], dtype=np.float64)
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env1 = ModifiedCartPoleEnv("state_norm")
    #state1 = env1.reset()
    #state2, reward, terminated, _, _ = env1.step(2)
    #print(state1, state2, reward, terminated)

    theta1 = []
    theta2 = []
 

    for j in range(1000):
        state1 = env1.reset()
        state = np.random.uniform(low=[-np.pi/2, -6], high=[np.pi/2, 6], size=(2,))
        env1.set_state(state)

        done = False

        while not done:
            theta1.append(state[1])

            action = env1.action_space.sample()
             
            state, reward, done, info, truncated = env1.step(action)
     
   
    print(f"Avg steps: {len(theta1)/1000}")
    plt.figure()
    #plt.plot(range(len(theta1)), theta1, label='Modified RK45 (solve_ivp)')
    _, bins, _ = plt.hist(theta1)
    #print(bins)

    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
          
