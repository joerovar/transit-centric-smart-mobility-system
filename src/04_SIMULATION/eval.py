from ins.Fixed_Inputs_81 import OUT_TRIP_RECORD_COLS, IN_TRIP_RECORD_COLS, PAX_RECORD_COLS
from sb3_contrib.ppo_mask import MaskablePPO
from run import BusEnv
import numpy as np
import os
import pandas as pd

def write_trip_records(save_folder, out_record_set, in_record_set, pax_record_set):
    os.mkdir(save_folder)
    path_out_trip_record = save_folder + '/trip_record_ob.pkl'
    path_in_trip_record = save_folder + '/trip_record_ib.pkl'
    path_pax_record = save_folder + '/pax_record_ob.pkl'

    out_trip_record = pd.concat(out_record_set, ignore_index=True)
    in_trip_record = pd.concat(in_record_set, ignore_index=True)
    pax_record = pd.concat(pax_record_set, ignore_index=True)

    out_trip_record.to_pickle(path_out_trip_record)
    in_trip_record.to_pickle(path_in_trip_record)
    pax_record.to_pickle(path_pax_record)
    return

def process_trip_record(record, record_col_names, rep_nr):
    df = pd.DataFrame(record, columns=record_col_names)
    df['replication'] = pd.Series([rep_nr + 1 for _ in range(len(df.index))])
    return df

def run_rl_scenario(episodes=1, cancelled_blocks=None, save_results=False, save_folder=None, messages=False, obs=None):
    config = {"HOLD_INTERVALS": 60,
              "IMPOSED_DELAY_LIMIT": 240}
    env = BusEnv(config, cancelled_blocks=cancelled_blocks)
    env.reset()

    model_path = "models/PPO_5am6pm/100000.zip"
    model = MaskablePPO.load(model_path, env=env)

    if obs:
        obs = np.array(obs, dtype=np.float32)
        action, _ = model.predict(obs)
        return action

    # episodes = 1
    all_rewards = []
    reward_list = []

    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        cnt = 0
        rewards = []
        while not done:
            action, _ = model.predict(obs)
            if messages:
                print("Observation: ", obs)
            obs, reward, done, info = env.step(action)
            if messages:
                print("Action: ", action)
                print("Reward: ", reward)
                print("Next observation: ", obs)
                print()
            cnt += 1
            rewards.append(reward)
        reward_list.append(rewards)
        all_rewards.append(np.mean(reward_list))
        if save_results:
            # record simulation results
            out_trip_record_set.append(process_trip_record(env.env.out_trip_record, OUT_TRIP_RECORD_COLS, ep))
            in_trip_record_set.append(process_trip_record(env.env.in_trip_record, IN_TRIP_RECORD_COLS, ep))
            pax_record_set.append(process_trip_record(env.env.completed_pax_record, PAX_RECORD_COLS, ep))
    if save_results:
        write_trip_records(save_folder, out_trip_record_set, in_trip_record_set, pax_record_set)
    env.close()
    # print("Average reward: ", np.mean(reward_list))

# print(run_rl_scenario(obs=[5*60, 5*60, 5*60, 5*60, 5*60, 5*60, 5*60, 5*60, 0*60]))
