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
        self.tau = 0.1  # seconds between state updates # 0.1
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

        theta, theta_dot = self.state
        #theta, theta_dot = self.state
        forces = [-self.force_mag, 0, self.force_mag]
    
        force_noise = np.random.uniform(-10, 10)

        #force_noise = np.random.normal(scale=5)
        force = forces[action] + force_noise
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        prev_state = self.state
        #print(prev_state)

        self.state = self.integrate(force)
        theta, _ = self.state
        
        #self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
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
      
        elif self.reward=="sq_angle": # REWARD SCHEMES FROM pendulum_simulator.m
            reward = (prev_state[0] * 2 / np.pi) ** 2 - (self.state[0] * 2 / np.pi) ** 2
        elif self.reward=="abs_angle_normalized":
            reward = np.abs(prev_state[0] * 2 / np.pi) - np.abs(self.state[0] * 2 / np.pi)
        elif self.reward=="abs_angle":
            reward = np.abs(prev_state[0]) - np.abs(self.state[0])
        elif self.reward=="state_norm":
            #print(prev_state)
            print(np.linalg.norm(self.state))
            reward = np.linalg.norm(prev_state) - np.linalg.norm(self.state)
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

        #self.state = np.array(self.state, dtype=np.float64)[[2,3]]
        return self.state
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

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
            '''
            temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            theta_acc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
            x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass
    
            return [x_dot, x_acc, theta_dot, theta_acc]
            '''
        
        sol = solve_ivp(cartpole_dynamics, [0, self.tau], self.state, method="RK23", t_eval=[self.tau])
        return np.array(sol.y[:, -1], dtype=np.float64)
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env1 = ModifiedCartPoleEnv("state_norm")
    state1 = env1.reset()
    state2, reward, terminated, _, _ = env1.step(2)
    print(state1, state2, reward, terminated)
    
    '''
    env2 = gym.make("CartPole-v1")

    theta1 = []
    theta2 = []
    for i in range(5):
        state = env1.reset(50)
        print(state)

    for j in range(100):
        state1 = env1.reset()
        state2 = env2.reset()
        state2 = state2[0]
        #if type(state) == tuple:
            #state = state[0]
        done1 = False
        done2 = False
        while not done1:
            theta1.append(state1[0])

            action = env1.action_space.sample()
             
            state1, reward, done1, info, truncated = env1.step(action)
     
        
        while not done2:
            theta2.append(state2[2])

            action = env2.action_space.sample()
             
            state2, reward, done2, info, truncated = env2.step(action)


    plt.figure()
    plt.plot(range(len(theta1)), theta1, label='Modified RK45 (solve_ivp)')
    plt.plot(range(len(theta2)), theta2, label='CartPole-v1 RK45 (solve_ivp)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    '''
    
          
