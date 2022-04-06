import argparse
import agents_dqn as Agents
from utils import plot_learning
import sim_env
from post_process import pax_per_trip_from_trajectory_set, load, save
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from input import STOPS, CONTROLLED_STOPS, FOCUS_TRIPS_MEAN_HW, N_ACTIONS_RL, LIMIT_HOLDING, IDX_RT_PROGRESS
from input import IDX_LOAD, IDX_PICK, IDX_DROP, N_STATE_PARAMS_RL, IDX_PAX_AT_STOP, IDX_FW_H, IDX_BW_H, CONTROL_MEAN_HW
import os

if __name__ == '__main__':
    tstamp_save = time.strftime("%m%d-%H%M")
    parser = argparse.ArgumentParser(
        description='Deep Q Learning: From Paper to Code')
    # the hyphen makes the argument optional
    parser.add_argument('-n_episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.01,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.985, help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1.5e-5, help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=0.64,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=8000, help='Maximum size for memory replay buffer')
    parser.add_argument('-bs', type=int, default=32, help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=600,
                        help='interval for replacing target network')
    parser.add_argument('-fc_dims', type=int, default=256, help='fully connected dimensions')
    parser.add_argument('-algo', type=str, default='DDQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-simple_reward', type=bool, default=False, help='delayed(false)/simple(true)')

    parser.add_argument('-weight_ride_t', type=float, default=0.0,
                        help='weight for ride time in reward')
    parser.add_argument('-tt_factor', type=float, default=1.0, help='dictates factor on variability')
    parser.add_argument('-hold_adj_factor', type=float, default=0.0, help='holding adjustment factor')
    parser.add_argument('-estimated_pax', type=float, default=False,
                        help='make stops be estimated for RL observations')

    parser.add_argument('-env', type=str, default=tstamp_save, help='environment')
    parser.add_argument('-load_checkpoint', type=bool, default=False, help='load model checkpoint')
    parser.add_argument('-model_base_path', type=str, default='out/trained_nets/',
                        help='path for model saving/loading')
    parser.add_argument('-test_save_folder', type=str, default=None, help='DDQN-LA/DDQN-HA')
    args = parser.parse_args()

    if not args.load_checkpoint:
        parent_dir = 'out/trained_nets/'
        child_dir = args.env
        path = os.path.join(parent_dir, child_dir)
        os.mkdir(path)

    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma, epsilon=args.eps, lr=args.lr, input_dims=[N_STATE_PARAMS_RL],
                   n_actions=N_ACTIONS_RL, mem_size=args.max_mem, eps_min=args.eps_min,
                   batch_size=args.bs, replace=args.replace, eps_dec=args.eps_dec,
                   chkpt_dir=args.model_base_path + args.env + '/', algo=args.algo,
                   env_name=args.env, fc_dims=args.fc_dims)
    if not args.load_checkpoint:
        # --------------------------------------------------- TRAINING -----------------------------------------
        figure_file = 'out/trained_nets/' + args.env + '/rew_curve.png'
        scores_file = 'out/trained_nets/' + args.env + '/rew_nums.csv'
        params_file = 'out/trained_nets/' + args.env + '/input_params.csv'

        arg_params = {'param': ['n_episodes', 'lr', 'eps_min', 'gamma', 'eps_dec', 'eps', 'max_mem', 'bs',
                                'replace', 'algo', 'simple_rew', 'fc_dims', 'weight_ride_time', 'limit_hold',
                                'tt_factor', 'hold_adj_factor', 'estimated_pax', 'n_episodes'],
                      'value': [args.n_episodes, args.lr, args.eps_min, args.gamma, args.gamma, args.eps, args.max_mem,
                                args.bs, args.replace, args.algo, args.simple_reward, args.fc_dims,
                                args.weight_ride_t, LIMIT_HOLDING, args.tt_factor, args.hold_adj_factor,
                                args.estimated_pax, args.n_episodes]}
        df_params = pd.DataFrame(arg_params)
        df_params.to_csv(params_file, index=False)

        scores, eps_history, steps = [], [], []
        n_steps = 0
        for j in range(args.n_episodes):
            score = 0
            env = sim_env.DetailedSimulationEnvWithDeepRL(hold_adj_factor=args.hold_adj_factor,
                                                          weight_ride_t=args.weight_ride_t)
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

                    if route_progress == 0.0 and abs(fw_h-bw_h) < 0.1*CONTROL_MEAN_HW:
                        action = agent.choose_action(observation, mask_idx=[0])
                        # print(action)
                    elif route_progress == 0.0 or pax_at_stop == 0 or previous_denied or args.simple_reward:
                        action = agent.choose_action(observation, mask_idx=[0])
                        # print(action)
                    else:
                        action = agent.choose_action(observation)
                    env.take_action(action)
                env.update_rewards(simple_reward=args.simple_reward)
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
                if not args.load_checkpoint:
                    agent.save_models()
                best_score = avg_score

            eps_history.append(agent.epsilon)
        plot_learning(steps, scores, eps_history, figure_file)
        scores_d = {'step': steps, 'score': scores, 'eps': eps_history}
        scores_df = pd.DataFrame(scores_d)
        scores_df.to_csv(scores_file, index=False)
    else:
        # --------------------------------- TESTING ----------------------------------------------------------------
        agent.load_models()
        tstamp = datetime.now().strftime('%m%d-%H%M%S')
        trajectories_set = []
        sars_set = []
        pax_set = []
        for j in range(args.n_episodes):
            score = 0
            env = sim_env.DetailedSimulationEnvWithDeepRL(tt_factor=args.tt_factor, hold_adj_factor=args.hold_adj_factor,
                                                          estimate_pax=args.estimated_pax)
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
                    if route_progress == 0.0 and abs(fw_h-bw_h) < 0.1*CONTROL_MEAN_HW:
                        action = agent.choose_action(observation, mask_idx=[0])
                        # print(action)
                    elif route_progress == 0.0 or pax_at_stop == 0 or previous_denied or args.simple_reward:
                        action = agent.choose_action(observation, mask_idx=[0])
                        # print(action)
                        # print(observation[0], observation[2], observation[3], action)
                    else:
                        action = agent.choose_action(observation)
                    env.take_action(action)
                done = env.prep()
            env.process_results()
            trajectories_set.append(env.trajectories)
            sars_set.append(env.trips_sars)
            pax_set.append(env.completed_pax)
        if args.test_save_folder:
            path_trajectories = 'out/' + args.test_save_folder + '/' + tstamp + '-trajectory_set.pkl'
            path_sars = 'out/' + args.test_save_folder + '/' + tstamp + '-sars_set.pkl'
            path_completed_pax = 'out/' + args.test_save_folder + '/' + tstamp + '-pax_set.pkl'
            # SAVE RESULTS
            save(path_trajectories, trajectories_set)
            save(path_sars, sars_set)
            save(path_completed_pax, pax_set)
            with open('out/' + args.test_save_folder + '/' + tstamp + '-net_used.csv', 'w') as f:
                f.write(str(args.env))

# cd src/04_SIMULATION
# train
# python main.py -n_episodes -simple_reward -weight_ride_t -hold_adj_factor
# test
# python main.py -n_episodes -eps -load_checkpoint -test_save_folder -simple_reward -hold_adj_factor -env (tstamp)

