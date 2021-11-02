import argparse
# import gym
import numpy as np
import agents as Agents
from utils import plot_learning
import simulation_env
import post_process
from file_paths import *
import output
from datetime import datetime
from constants import *

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
    parser.add_argument('-eps_dec', type=float, default=1e-4,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=10000,
                        help='Maximum size for memory replay buffer')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=800,
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
                   input_dims=[4],
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

        tstamp_save = time.strftime("%m%d-%H%M")
        fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' +\
            str(args.n_games) + 'eps_' + tstamp_save

        figure_file = 'out/training plots/' + fname + '.png'
        scores_file = fname + '_scores.npy'

        scores, eps_history, steps_array = [], [], []
        n_steps = 0
        for j in range(args.n_games):
            score = 0
            env = simulation_env.SimulationEnvDeepRL()
            done = env.reset_simulation()
            done = env.prep()
            while not done:
                i = env.bus_idx
                trip_id = env.active_trips[i]
                all_sars = env.trips_sars[trip_id]

                if len(all_sars) > 1:
                    if env.bool_terminal_state:
                        prev_sars = all_sars[-1]
                    else:
                        prev_sars = all_sars[-2]
                    observation, action, reward, observation_ = prev_sars
                    observation = np.array(observation, dtype=np.float32)
                    observation_ = np.array(observation_, dtype=np.float32)
                    score += reward
                    agent.store_transition(observation, action, reward, observation_, int(env.bool_terminal_state))
                    agent.learn()

                if not env.bool_terminal_state:
                    observation = np.array(all_sars[-1][0], dtype=np.float32)
                    action = agent.choose_action(observation)
                    env.take_action(action)

                done = env.prep()
                n_steps += 1
            if not args.load_checkpoint:
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
    # if args.load_checkpoint and n_steps >= 18000:
    #     break
    else:
        agent.load_models()
        tstamps = []
        for j in range(args.n_games):
            score = 0
            env = simulation_env.SimulationEnvDeepRL()
            done = env.reset_simulation()
            done = env.prep()
            while not done:
                if not env.bool_terminal_state:
                    i = env.bus_idx
                    trip_id = env.active_trips[i]
                    all_sars = env.trips_sars[trip_id]
                    observation = np.array(all_sars[-1][0], dtype=np.float32)
                    action = agent.choose_action(observation)
                    env.take_action(action)
                done = env.prep()

            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-4])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            path_sars = path_to_outs + dir_var + 'sars_record_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
            post_process.save(path_sars, env.trips_sars)
        # output.get_results()
        output.get_rl_results(tstamps)

