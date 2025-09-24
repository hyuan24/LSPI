import string
import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
import math

import gym
import numpy as np

"""
This implementation is modified from http://incompleteideas.net/sutton/book/code/pole.c, 
which corresponds to the cart-pole system as described in Barto et. al. 
- The action space has an additional action (do nothing) from 
the gymnasium version; additonally, the force applied has Gaussian noise. 
- There are two sets of parameters: one corresponding to the orignal gymnasium set, one following the 
problem description in "Least-Squares Policy Iteration" (Lagoudakis and Parr) @ https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf 
- The action space is Discrete(3) with actions Left (0), Do nothing (1), Right (2)
- The observation space is continuous with shape (2,). While the environment internally keeps track of position,
horizontal velocity, angle, and angular velocity, step() and reset() return observations with only angle and angular 
velocity (since only those are used in Lagoudakis)

Cart Position allowed range before termination is (-2.4, 2.4) meters ("gym") OR (-4.8, 4.8) ("lagoudakis")
Pole Angle (from upright) range is (-0.2095, 0.2095) radians ("gym") OR (-pi/2, pi/2) ("lagoudakis")
Episodes are not terminated based on cart velocity or pole angular velocity. 

Episodes start with all observations between (-0.05, 0.05).
"""

class CartPole3(gym.Env):
    
    def __init__(
        self, param: string = "gym", sutton_barto_reward: bool = False, render_mode: str | None = None
    ):
        self._sutton_barto_reward = sutton_barto_reward
        self.param = param
        self.gravity = 9.8
        self.kinematics_integrator = "euler"

        if self.param == "gym":
            self.masscart = 1.0
            self.masspole = 0.1
            self.length = 0.5  # actually half the pole's length 
            self.force_mag = 10.0
            self.tau = 0.02  # seconds between state update
            self.theta_threshold_radians = 12 * 2 * math.pi / 360 # Angle at which to fail the episode
            self.x_threshold = 2.4
            self.noise_factor = 2
        elif self.param == "lagoudakis":
            self.masscart = 8.0 
            self.masspole = 2.0 
            self.length = 0.5 
            self.force_mag = 50.0 
            self.tau = 0.1  # seconds between state updates 
            self.theta_threshold_radians = math.pi/2
            self.x_threshold = np.inf
            self.noise_factor = self.force_mag / 5
        else: ValueError("Invalid param set.")

        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

 
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

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
        x, x_dot, theta, theta_dot = self.state

        forces = [-self.force_mag, 0, self.force_mag]
        force = forces[action] + np.random.uniform(-10, 10) 
        #force = forces[action] + np.random.normal() * self.noise_factor
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32)[[2,3]], reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32)[[2,3]]


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
'''
class CartPoleWrapper(gym.Env):
    def __init__(self):
        super().__init__()

        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.param = "gym"
    def step(self, action):
        assert action in self.action_space
        state, reward, done, _, _ = self.env.step(action)
        return state[[2,3]], reward, done, False, {}
    def reset(self, seed = None, options = None):
        return self.env.reset(seed=seed, options=options)[0][[2,3]]
'''

    

