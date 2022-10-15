from sb3_contrib.ppo_mask import MaskablePPO
from run import BusEnv
import numpy as np

if __name__=="__main__":
    config = {"HOLD_INTERVALS": 60,
              "IMPOSED_DELAY_LIMIT": 240}
    env = BusEnv(config)
    env.reset()

    # load the trained model
    model_path = "models/PPO/200000.zip"
    model = MaskablePPO.load(model_path)

    episodes = 1
    all_rewards = []
    reward_list = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        cnt = 0
        rewards = []
        while not done:
            # interact with env
            action, _ = model.predict(obs, action_masks=env.action_masks())
            print("Observation: ", obs)
            obs, reward, done, info = env.step(action)
            print("Action: ", action)
            print("Reward: ", reward)
            print("Next observation: ", obs)
            print()
            cnt += 1
            rewards.append(reward)
        reward_list.append(rewards)
        all_rewards.append(np.mean(reward_list))
    env.close()
    print("Average reward: ", np.mean(reward_list))
