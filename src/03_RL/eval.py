import argparse
from typing import Union

from pathlib import Path
from sb3_contrib.ppo_mask import MaskablePPO

import gym
from gym import spaces
import numpy as np

class TestEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config=None, cancelled_blocks=None):

        # define action & observation space
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(9,), dtype=np.float32)
        self.invalid_actions = []
        self.possible_actions = list(range(self.action_space.n))


    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        return self.observation_space.sample()  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def action_masks(self):
        return [action not in self.invalid_actions for action in self.possible_actions]


def load_model(model_name: str,
               env: gym.Env,
               path: Path = Path("models/")
               ) -> MaskablePPO:
    model_path = Path(path) / model_name
    model = MaskablePPO.load(model_path, env=env)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--path", type=str, default="models/")
    args = parser.parse_args()

    env = TestEnv()
    model = load_model(args.model_name, env, args.path)
    
    episodes = 10
    reward_list = []
    cnt = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        cnt = 0
        rewards = []
        while not done:
            cnt += 1
            if cnt > 1000:
                break
            action, _ = model.predict(obs)
            print("Observation: ", obs)
            obs, reward, done, info = env.step(action)
            print("Action: ", action)
            print("Reward: ", reward)
            print("Next observation: ", obs)
            print()
            cnt += 1
            rewards.append(reward)
        reward_list.append(rewards)
    env.close()