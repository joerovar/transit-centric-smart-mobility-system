'''
To do:
1, normalize state space
2, parameter tuning
'''

import numpy as np
import os
import gym
from gym import spaces

from params import *

from SimEnvs import SimulationEnvWithCancellations

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from sb3_contrib import TRPO
from Variable_Inputs import BLOCK_IDS
import optuna


class BusEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config, cancelled_blocks=None):
        super(BusEnv, self).__init__()
        # wrap environment
        self.env = SimulationEnvWithCancellations(control_strategy='RL', weight_hold_t=0,
                                                    cancelled_blocks=cancelled_blocks)
        self.env.reset_simulation()
        self.env.prep()

        # define action & observation space
        self.action_space = spaces.Discrete(NR_ACTIONS_D_RL)
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(9,), dtype=np.float32)
        self.invalid_actions = []
        self.possible_actions = list(range(self.action_space.n))

        # set parameters
        self.HOLD_INTERVALS = config["HOLD_INTERVALS"]
        self.IMPOSED_DELAY_LIMIT = config["IMPOSED_DELAY_LIMIT"]

    def step(self, action):
        self.env.dispatch_decision(hold_time=action * self.HOLD_INTERVALS)
        done = self.env.prep()
        obs, reward = np.array(self.env.obs, dtype=np.float32), self.env.prev_reward

        # find invalid actions
        if not done:
            hold_t_max = max(self.IMPOSED_DELAY_LIMIT - obs[-1], 0)
            max_action = int(hold_t_max / self.HOLD_INTERVALS)
            self.invalid_actions = [action for action in self.possible_actions if action > max_action]

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.invalid_actions = []
        # p = np.random.uniform(0.0,0.25)
        # cancelled_blocks = np.random.choice(BLOCK_IDS, replace=False, size=int(p*len(BLOCK_IDS))).tolist()
        # self.env = SimulationEnvWithCancellations(control_strategy='RL', weight_hold_t=0,
        #                                     cancelled_blocks=cancelled_blocks)
        self.env.reset_simulation()
        
        self.env.prep()
        return np.array(self.env.obs, dtype=np.float32)  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def action_masks(self):
        return [action not in self.invalid_actions for action in self.possible_actions]


# def mask_fn(env):
#     return [action not in env.invalid_actions for action in env.possible_actions]

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'n_steps': pow(2, int(trial.suggest_uniform('n_steps', 6, 11))),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-2),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 1.),
        'batch_size': 64
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    config = {"HOLD_INTERVALS": HOLD_INTERVALS,
              "IMPOSED_DELAY_LIMIT": IMPOSED_DELAY_LIMIT}
    model_params = optimize_ppo(trial)
    print(model_params["n_steps"])
    env = BusEnv(config)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=0, **model_params)
    model.learn(20000)
    mean_reward, _ = evaluate_policy(model, BusEnv(config), n_eval_episodes=10)
    env.close()

    return mean_reward



if __name__ == '__main__':
    model_dir = "models/PPO_haris"
    logdir = "logs"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # model_params={'clip_range': 0.3703600835330346,
    #               'ent_coef': 0.0002596976573399199,
    #               'gae_lambda': 0.8794093959952141,
    #               'gamma': 0.905537653956815,
    #               'learning_rate': 0.00013105202351342582,
    #               'n_steps': pow(2, 8)}
    # model_params = {}
    config = {"HOLD_INTERVALS": HOLD_INTERVALS,
              "IMPOSED_DELAY_LIMIT": IMPOSED_DELAY_LIMIT}
    env = BusEnv(config)
    # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=0, **model_params, tensorboard_log=logdir)
    # model.learn(20000, tb_log_name="Best")
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=logdir)
    for i in range(N_EPISODES_TRAIN):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO(final)")
        model.save(f"{model_dir}/{TIMESTEPS * (i+1)}")

    # study = optuna.create_study(direction="maximize", study_name="bus-control", storage="sqlite:///bus-control.db")
    # try:
    #     study.optimize(optimize_agent, n_trials=200, n_jobs=4)
    # except KeyboardInterrupt:
    #     print('Interrupted by keyboard.')

# if __name__ == "__main__":
#     model_dir = "models/PPO"
#     logdir = "logs"
#
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)
#
#     config = {"HOLD_INTERVALS": HOLD_INTERVALS,
#               "IMPOSED_DELAY_LIMIT": IMPOSED_DELAY_LIMIT}
#     env = BusEnv(config)
#     check_env(env)
#     model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=logdir)
#     # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
#
#     for i in range(N_EPISODES_TRAIN):
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
#         model.save(f"{model_dir}/{TIMESTEPS * (i+1)}")
#     # config = {"HOLD_INTERVALS": 30}
#     # env = CustomEnv(config)
