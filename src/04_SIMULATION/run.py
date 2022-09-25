import numpy as np
import gym
from gym import spaces

from Simulation_Processor import rl_dispatch
from SimEnvs import SimulationEnvWithCancellations

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


N_EPISODES_TRAIN = 1
DEFAULT_ACTION = 0
HOLD_INTERVALS = 30


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config):
        super(CustomEnv, self).__init__()
        # wrap environment
        self.env = SimulationEnvWithCancellations(control_strategy='RL', weight_hold_t=0)
        self.env.reset_simulation()
        self.env.prep()

        # define action & observation space
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(9,), dtype=np.float32)

        # set parameters
        self.HOLD_INTERVALS = config["HOLD_INTERVALS"]

    def step(self, action):
        self.env.dispatch_decision(hold_time=action * self.HOLD_INTERVALS)
        done = self.env.prep()
        obs, reward = np.array(self.env.obs, dtype=np.float32), self.env.prev_reward

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.env.reset_simulation()
        self.env.prep()
        return np.array(self.env.obs, dtype=np.float32) # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    config = {"HOLD_INTERVALS": 30}
    env = CustomEnv(config)
    # check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)