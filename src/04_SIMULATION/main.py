import argparse
# import gym
import numpy as np
import agents as Agents
from utils import plot_learning
import simulation_env
import post_process
from file_paths import *
from datetime import datetime
from constants import *
import matplotlib.pyplot as plt
from input import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deep Q Learning: From Paper to Code')
    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.01,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1.4e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=0.6,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=8000,
                        help='Maximum size for memory replay buffer')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=600,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='BusControl',
                        help='environment')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='out/models/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    args = parser.parse_args()

    # env = gym.make(args.env)
    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                   epsilon=args.eps,
                   lr=args.lr,
                   input_dims=[N_STATE_PARAMS_RL],
                   n_actions=N_ACTIONS_RL,
                   mem_size=args.max_mem,
                   eps_min=args.eps_min,
                   batch_size=args.bs,
                   replace=args.replace,
                   eps_dec=args.eps_dec,
                   chkpt_dir=args.path,
                   algo=args.algo,
                   env_name=args.env)
    if not args.load_checkpoint:

        # --------------------------------------------------- TRAINING -----------------------------------------

        tstamp_save = time.strftime("%m%d-%H%M")
        fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' + str(args.n_games) + 'eps_' + tstamp_save

        figure_file = 'out/training plots/' + fname + '.png'
        scores_file = fname + '_scores.npy'

        scores, eps_history, steps_array = [], [], []
        n_steps = 0
        for j in range(args.n_games):
            score = 0
            env = simulation_env.DetailedSimulationEnvWithDeepRL()
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
                        action = agent.choose_action(observation, mask_idx=0)
                    else:
                        action = agent.choose_action(observation)
                    env.take_action(action)
                env.update_rewards()
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
            # nr_valuable_experiences = len(env.pool_sars)
            # for i in range(nr_valuable_experiences):
            #     observation, action, reward, observation_, terminal = env.pool_sars[i]
            #     observation = np.array(observation, dtype=np.float32)
            #     observation_ = np.array(observation_, dtype=np.float32)
            #     score += reward
            #     terminal = int(terminal)
            #     agent.store_transition(observation, action, reward, observation_, terminal)
            #     agent.learn()
            # n_steps += nr_valuable_experiences
            scores.append(score)
            steps_array.append(n_steps)
            avg_score = np.mean(scores[-100:])
            print('episode ', j, 'score %.2f' % score, 'average score %.2f' % avg_score,
                  'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
            if avg_score > best_score:
                if not args.load_checkpoint:
                    agent.save_models()
                best_score = avg_score

            eps_history.append(agent.epsilon)
        plot_learning(steps_array, scores, eps_history, figure_file)
    else:
        # --------------------------------- TESTING ----------------------------------------------------------------

        agent.load_models()
        # tstamp = datetime.now().strftime('%m%d-%H%M%S')
        # trajectories_set = []
        # sars_set = []
        # pax_set = []
        # for j in range(args.n_games):
        #     score = 0
        #     env = simulation_env.DetailedSimulationEnvWithDeepRL()
        #     done = env.reset_simulation()
        #     done = env.prep()
        #     while not done:
        #         if not env.bool_terminal_state:
        #             trip_id = env.bus.active_trip[0].trip_id
        #             all_sars = env.trips_sars[trip_id]
        #             observation = np.array(all_sars[-1][0], dtype=np.float32)
        #             route_progress = observation[IDX_RT_PROGRESS]
        #             pax_at_stop = observation[IDX_PAX_AT_STOP]
        #
        #             curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
        #             previous_denied = False
        #             for p in curr_stop[0].pax.copy():
        #                 if p.arr_time <= env.time:
        #                     if p.denied:
        #                         previous_denied = True
        #                         break
        #                 else:
        #                     break
        #
        #             if route_progress == 0.0 or pax_at_stop == 0 or previous_denied:
        #                 action = agent.choose_action(observation, mask_idx=0)
        #             else:
        #                 action = agent.choose_action(observation)
        #             env.take_action(action)
        #         done = env.prep()
        #     env.process_results()
        #     trajectories_set.append(env.trajectories)
        #     sars_set.append(env.trips_sars)
        #     pax_set.append(env.completed_pax)
        #
        # path_trajectories = 'out/RL/trajectory_set_' + tstamp + '.pkl'
        # path_sars = 'out/RL/sars_set_' + tstamp + '.pkl'
        # path_completed_pax = 'out/RL/pax_set_' + tstamp + '.pkl'
        # post_process.save(path_trajectories, trajectories_set)
        # post_process.save(path_sars, sars_set)
        # post_process.save(path_completed_pax, pax_set)

        # trajectories_set = load('out/RL/trajectory_set_0106-191406.pkl')
        # load_mean, _, _, ons_mean, _ = pax_per_trip_from_trajectory_set(trajectories_set, IDX_LOAD,
        #                                                                 IDX_PICK, IDX_DROP, STOPS)
        #
        # stable = (FOCUS_TRIPS_MEAN_HW, FOCUS_TRIPS_MEAN_HW, FOCUS_TRIPS_MEAN_HW)
        # early = (FOCUS_TRIPS_MEAN_HW-FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW+FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW)
        # late = (FOCUS_TRIPS_MEAN_HW+FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW-FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW)
        # policies = np.zeros((3, len(CONTROLLED_STOPS)-1, N_ACTIONS_RL))
        # route_progress_scenarios = np.array([(STOPS.index(s) / len(STOPS)) for s in CONTROLLED_STOPS[:-1]])
        # avg_load = np.array([load_mean[STOPS.index(s)] for s in CONTROLLED_STOPS[:-1]])
        # avg_ons = np.array([ons_mean[STOPS.index(s)] for s in CONTROLLED_STOPS[:-1]])
        # k = 0
        # for scenario in (stable, early, late):
        #     for i in range(route_progress_scenarios.size):
        #         obs = np.array([route_progress_scenarios[i], avg_load[i], scenario[0], scenario[1], avg_ons[i], scenario[2]], dtype=np.float32)
        #         q_values = agent.q_values(obs).detach().numpy()[0]
        #         temp = np.argsort(q_values)
        #         rank_q_values = np.empty_like(temp)
        #         rank_q_values[temp] = np.arange(len(q_values))
        #         policies[k, i] = rank_q_values
        #     k += 1
        # fig_names = ['on time', 'early', 'late']
        # for i in range(policies.shape[0]):
        #     plt.imshow(policies[i].T, cmap='Greens')
        #     plt.xticks(np.arange(len(route_progress_scenarios)), np.arange(len(route_progress_scenarios)) + 1, fontsize=8)
        #     plt.yticks(np.arange(N_ACTIONS_RL), ['skip', 'no hold', 'hold 25s', 'hold 50s', 'hold 75s', 'hold 100s'], fontsize=8)
        #     plt.xlabel('control stop', fontsize=8)
        #     plt.title('policy for ' + fig_names[i] + ' bus')
        #     plt.savefig('policy_' + fig_names[i] + '.png')
        #     plt.close()
        # action_grid = np.zeros(shape=(len(fw_headway_scenarios), len(route_progress_scenarios)))
        #
        # for i in range(fw_headway_scenarios.size):
        #     for j in range(route_progress_scenarios.size):
        #         obs = np.array([route_progress_scenarios[j], int(load_mean[CONTROLLED_STOPS[j]]),
        #                         fw_headway_scenarios[i], bw_headway_scenarios[i],
        #                         int(ons_mean[CONTROLLED_STOPS[j]])], dtype=np.float32)
        #         action_grid[i, j] = agent.choose_action(obs)
        #
        # fig, ax = plt.subplots()
        # x_axis = np.arange(route_progress_scenarios.size)
        # y_axis = np.arange(fw_headway_scenarios.size)
        # ms = ax.matshow(action_grid)
        # ax.set_xticks(x_axis)
        # ax.xaxis.set_ticks_position('bottom')
        # ax.set_xticklabels(np.arange(1, route_progress_scenarios.size + 1))
        # ax.set_yticks(y_axis)
        # ax.set_yticklabels(np.arange(60, -80, -20))
        # ax.set_xlabel('control point')
        # ax.set_ylabel('dfh=-dbh (seconds)')
        # cbar = fig.colorbar(ms, ticks=np.arange(np.min(action_grid), np.max(action_grid) + 1), orientation='horizontal')
        # cbar.ax.set_xlabel('best action')
        # path_policy_examination = path_to_outs + 'RL/' + 'policy_' + args.env + ext_fig
        # plt.savefig(path_policy_examination)

# cd src/04_SIMULATION
# train
# python main.py -env -algo -n_games
# test
# python main.py -env -algo -n_games -eps -load_checkpoint
