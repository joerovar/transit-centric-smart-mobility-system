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
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=9e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=10000,
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
                   input_dims=[5],
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
        fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' + \
                str(args.n_games) + 'eps_' + tstamp_save

        figure_file = 'out/training plots/' + fname + '.png'
        scores_file = fname + '_scores.npy'

        scores, eps_history, steps_array = [], [], []
        n_steps = 0
        for j in range(args.n_games):
            score = 0
            env = simulation_env.DetailedSimulationEnvWithDeepRL()
            done = env.reset_simulation()
            done = env.prep()
            # temp_n_steps = 0
            while not done:
                trip_id = env.bus.active_trip[0].trip_id
                all_sars = env.trips_sars[trip_id]
                # if len(all_sars) > 1:
                #     if env.bool_terminal_state:
                #         prev_sars = all_sars[-1]
                #     else:
                #         prev_sars = all_sars[-2]
                    # observation, action, reward, observation_ = prev_sars
                    # observation = np.array(observation, dtype=np.float32)
                    # observation_ = np.array(observation_, dtype=np.float32)
                    # score += reward
                    # agent.store_transition(observation, action, reward, observation_, int(env.bool_terminal_state))
                if not env.bool_terminal_state:
                    observation = np.array(all_sars[-1][0], dtype=np.float32)
                    action = agent.choose_action(observation)
                    env.take_action(action)
                env.update_rewards()
                # temp_n_steps += 1
                done = env.prep()
            nr_valuable_experiences = len(env.pool_sars)
            for i in range(nr_valuable_experiences):
                observation, action, reward, observation_, terminal = env.pool_sars[i]
                observation = np.array(observation, dtype=np.float32)
                observation_ = np.array(observation_, dtype=np.float32)
                score += reward
                terminal = int(terminal)
                agent.store_transition(observation, action, reward, observation_, terminal)
                agent.learn()
            n_steps += nr_valuable_experiences
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
        tstamp = datetime.now().strftime('%m%d-%H%M%S')
        trajectories_set = []
        sars_set = []
        pax_set = []
        for j in range(args.n_games):
            score = 0
            env = simulation_env.DetailedSimulationEnvWithDeepRL()
            done = env.reset_simulation()
            done = env.prep()
            while not done:
                if not env.bool_terminal_state:
                    trip_id = env.bus.active_trip[0].trip_id
                    all_sars = env.trips_sars[trip_id]
                    observation = np.array(all_sars[-1][0], dtype=np.float32)
                    action = agent.choose_action(observation)
                    env.take_action(action)
                done = env.prep()
            env.process_results()
            trajectories_set.append(env.trajectories)
            sars_set.append(env.trips_sars)
            pax_set.append(env.completed_pax)

        path_trajectories = 'out/RL/trajectory_set_' + tstamp + '.pkl'
        path_sars = 'out/RL/sars_set_' + tstamp + '.pkl'
        path_completed_pax = 'out/RL/pax_set_' + tstamp + '.pkl'
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_sars, sars_set)
        post_process.save(path_completed_pax, pax_set)

        # load_mean, _, ons_mean, _, _, _ = pax_per_trip_from_trajectory_set(trajectories_set, IDX_LOAD,
        #                                                                    IDX_PICK, IDX_DROP)

        # fw_headway_scenarios = np.arange(HEADWAY_UNIFORM + 60, HEADWAY_UNIFORM - 80, -20)
        # bw_headway_scenarios = np.flip(fw_headway_scenarios, axis=0)
        # route_progress_scenarios = np.array([(STOPS.index(s) / len(STOPS)) for s in CONTROLLED_STOPS[:-1]])
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
