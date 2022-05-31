import random
import agents_dqn as Agents
from utils import plot_learning
import sim_env
import numpy as np
import time
import pandas as pd
from input import N_ACTIONS_RL, LIMIT_HOLDING, CONTROL_MEAN_HW
import os
from constants import *


def train(n_episodes_train, simple_reward = False):
    tstamp_policy = time.strftime("%m%d-%H%M")
    agent_ = getattr(Agents, ALGO)
    agent = agent_(gamma=DISCOUNT_FACTOR, epsilon=EPS, lr=LEARN_RATE,
                   input_dims=[N_STATE_PARAMS_RL], n_actions=N_ACTIONS_RL,
                   mem_size=MAX_MEM, eps_min=EPS_MIN, batch_size=BATCH_SIZE, replace=EPISODES_REPLACE, eps_dec=EPS_DEC,
                   chkpt_dir=NETS_PATH + tstamp_policy + '/', algo=ALGO, env_name=tstamp_policy, fc_dims=FC_DIMS)
    best_score = -np.inf
    parent_dir = 'out/trained_nets/'
    child_dir = tstamp_policy
    path = os.path.join(parent_dir, child_dir)
    os.mkdir(path)
    figure_file = 'out/trained_nets/' + tstamp_policy + '/rew_curve.png'
    scores_file = 'out/trained_nets/' + tstamp_policy + '/rew_nums.csv'
    params_file = 'out/trained_nets/' + tstamp_policy + '/input_params.csv'

    arg_params = {'param': ['n_episodes', 'lr', 'eps_min', 'gamma', 'eps_dec', 'eps', 'max_mem', 'bs',
                            'replace', 'algo', 'simple_rew', 'fc_dims', 'weight_ride_time', 'limit_hold',
                            'tt_factor', 'hold_adj_factor', 'estimated_pax'],
                  'value': [n_episodes_train, LEARN_RATE, EPS_MIN, DISCOUNT_FACTOR, EPS_DEC, EPS, MAX_MEM,
                            BATCH_SIZE, EPISODES_REPLACE, ALGO, simple_reward, FC_DIMS,
                            WEIGHT_RIDE_T, LIMIT_HOLDING, TT_FACTOR, HOLD_ADJ_FACTOR, ESTIMATED_PAX]}
    df_params = pd.DataFrame(arg_params)
    df_params.to_csv(params_file, index=False)

    scores, eps_history, steps = [], [], []
    n_steps = 0
    for j in range(n_episodes_train):
        score = 0
        env = sim_env.DetailedSimulationEnvWithDeepRL(hold_adj_factor=HOLD_ADJ_FACTOR,
                                                      weight_ride_t=WEIGHT_RIDE_T)
        done = env.reset_simulation()
        done = env.prep()
        nr_sars_stored = 0
        while not done:
            trip_id = env.bus.active_trip[0].trip_id
            all_sars = env.trips_sars[trip_id]
            if not env.bool_terminal_state:
                observation = np.array(all_sars[-1][0], dtype=np.float32)
                route_progress = observation[IDX_RT_PROGRESS]
                pax_at_stop = observation[IDX_PAX_AT_STOP]
                fw_h = observation[IDX_FW_H]
                bw_h = observation[IDX_BW_H]

                curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
                previous_denied = False
                for p in curr_stop[0].pax.copy():
                    if p.arr_time <= env.time:
                        if p.denied:
                            previous_denied = True
                            break
                    else:
                        break

                if route_progress == 0.0 and abs(fw_h - bw_h) < 0.1 * CONTROL_MEAN_HW:
                    action = agent.choose_action(observation, mask_idx=[0])
                elif route_progress == 0.0 or pax_at_stop == 0 or previous_denied or simple_reward:
                    action = agent.choose_action(observation, mask_idx=[0])
                else:
                    action = agent.choose_action(observation)
                env.take_action(action)
            env.update_rewards(simple_reward=simple_reward)
            if len(env.pool_sars) > nr_sars_stored:
                observation, action, reward, observation_, terminal = env.pool_sars[-1]
                observation = np.array(observation, dtype=np.float32)
                observation_ = np.array(observation_, dtype=np.float32)
                score += reward
                terminal = int(terminal)
                agent.store_transition(observation, action, reward, observation_, terminal)
                agent.learn()
                nr_sars_stored += 1
                n_steps += 1
            done = env.prep()
        scores.append(score)
        steps.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('episode ', j, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
        if avg_score > best_score:
            agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
    plot_learning(steps, scores, eps_history, figure_file)
    scores_d = {'step': steps, 'score': scores, 'eps': eps_history}
    scores_df = pd.DataFrame(scores_d)
    scores_df.to_csv(scores_file, index=False)
    return


def test(n_episodes_test, tstamp_policy, test_save_folder='DDQN-HA',save_results=False, simple_reward = False):
    agent_ = getattr(Agents, ALGO)
    agent = agent_(gamma=DISCOUNT_FACTOR, epsilon=EPS, lr=LEARN_RATE, input_dims=[N_STATE_PARAMS_RL], n_actions=N_ACTIONS_RL,
                   mem_size=MAX_MEM, eps_min=EPS_MIN, batch_size=BATCH_SIZE, replace=EPISODES_REPLACE, eps_dec=EPS_DEC,
                   chkpt_dir=NETS_PATH + tstamp_policy + '/', algo=ALGO, env_name=tstamp_policy, fc_dims=FC_DIMS)
    agent.load_models()
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []
    for j in range(n_episodes_test):
        env = sim_env.DetailedSimulationEnvWithDeepRL(tt_factor=TT_FACTOR, hold_adj_factor=HOLD_ADJ_FACTOR,
                                                      estimate_pax=ESTIMATED_PAX)
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            if not env.bool_terminal_state:
                trip_id = env.bus.active_trip[0].trip_id
                all_sars = env.trips_sars[trip_id]
                observation = np.array(all_sars[-1][0], dtype=np.float32)
                route_progress = observation[IDX_RT_PROGRESS]
                pax_at_stop = observation[IDX_PAX_AT_STOP]
                fw_h = observation[IDX_FW_H]
                bw_h = observation[IDX_BW_H]
                curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
                previous_denied = False
                for p in curr_stop[0].pax.copy():
                    if p.arr_time <= env.time:
                        if p.denied:
                            previous_denied = True
                            break
                    else:
                        break
                if route_progress == 0.0 and abs(fw_h - bw_h) < 0.1 * CONTROL_MEAN_HW:
                    action = agent.choose_action(observation, mask_idx=[0])
                elif route_progress == 0.0 or pax_at_stop == 0 or previous_denied or simple_reward:
                    action = agent.choose_action(observation, mask_idx=[0])
                else:
                    action = agent.choose_action(observation)
                env.take_action(action)
            done = env.prep()

        if save_results:
            out_trip_record_df = pd.DataFrame(env.out_trip_record, columns=OUT_TRIP_RECORD_COLS)
            out_trip_record_df['replication'] = pd.Series([j+1 for _ in range(len(out_trip_record_df.index))])
            out_trip_record_set.append(out_trip_record_df)

            in_trip_record_df = pd.DataFrame(env.in_trip_record, columns=IN_TRIP_RECORD_COLS)
            in_trip_record_df['replication'] = pd.Series([j+1 for _ in range(len(in_trip_record_df.index))])
            in_trip_record_set.append(in_trip_record_df)

            pax_record_df = pd.DataFrame(env.completed_pax_record, columns=PAX_RECORD_COLS)
            pax_record_df['replication'] = pd.Series([j+1 for _ in range(len(in_trip_record_df.index))])
            pax_record_set.append(pax_record_df)
    if save_results:
        path_out_trip_record = 'out/' + test_save_folder + '/' + tstamp + '-trip_record_outbound.pkl'
        path_in_trip_record = 'out/' + test_save_folder + '/' + tstamp + '-trip_record_inbound.pkl'
        path_pax_record = 'out/' + test_save_folder + '/' + tstamp + '-pax_record_outbound.pkl'

        out_trip_record = pd.concat(out_trip_record_set, ignore_index=True)
        in_trip_record = pd.concat(in_trip_record_set, ignore_index=True)
        pax_record = pd.concat(pax_record_set, ignore_index=True)

        out_trip_record.to_pickle(path_out_trip_record)
        in_trip_record.to_pickle(path_in_trip_record)
        pax_record.to_pickle(path_pax_record)

        with open('out/' + test_save_folder + '/' + tstamp + '-net_used.csv', 'w') as f:
            f.write(str(tstamp_policy))
    return


def run_sample_rl(episodes=1, simple_reward=False, weight_ride_t=0.0):
    for _ in range(episodes):
        env = sim_env.DetailedSimulationEnvWithDeepRL(estimate_pax=True, weight_ride_t=weight_ride_t)
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            trip_id = env.bus.active_trip[0].trip_id
            all_sars = env.trips_sars[trip_id]
            if not env.bool_terminal_state:
                observation = np.array(all_sars[-1][0], dtype=np.float32)
                route_progress = observation[IDX_RT_PROGRESS]
                pax_at_stop = observation[IDX_PAX_AT_STOP]
                curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
                previous_denied = False
                for p in curr_stop[0].pax.copy():
                    if p.arr_time <= env.time:
                        if p.denied:
                            previous_denied = True
                            break
                    else:
                        break
                if route_progress == 0.0 or pax_at_stop == 0 or previous_denied:
                    action = random.randint(1, 4)
                else:
                    action = random.randint(0, 4)
                env.take_action(action)
            env.update_rewards(simple_reward=simple_reward)
            done = env.prep()
    return


