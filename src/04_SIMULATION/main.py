import argparse
# import gym
import numpy as np
import agents as Agents
from utils import plot_learning
import simulation_env
import post_process
from file_paths import *
import output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Deep Q Learning: From Paper to Code')
    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=5e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=50000,
                        help='Maximum size for memory replay buffer')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=500,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='BusControl',
                        help='environment')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/',
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
                   input_dims=[3],
                   n_actions=5,
                   mem_size=args.max_mem,
                   eps_min=args.eps_min,
                   batch_size=args.bs,
                   replace=args.replace,
                   eps_dec=args.eps_dec,
                   chkpt_dir=args.path,
                   algo=args.algo,
                   env_name=args.env)

    if args.load_checkpoint:
        agent.load_models()

    fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' +\
        str(args.n_games) + 'games'

    figure_file = 'plots/' + fname + '.png'
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
                if not args.load_checkpoint:
                    prev_sars = all_sars[-2]
                    observation, action, reward, observation_ = prev_sars
                    observation = np.array(observation, dtype=np.float32)
                    observation_ = np.array(observation_, dtype=np.float32)
                    score += reward
                    agent.store_transition(observation, action, reward, observation_, int(done))
                    agent.learn()
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
        if args.load_checkpoint and n_steps >= 18000:
            break
        if args.load_checkpoint:
            env.process_results()
            post_process.save(path_tr_save, env.trajectories)
            post_process.save(path_sars_save, env.trips_sars)
            output.get_results()
            output.get_rl_results()
    if not args.load_checkpoint:
        plot_learning(steps_array, scores, eps_history, figure_file)


    # for i in range(args.n_games):
    #     score = 0
    #     done = False
    #     observation = env.reset()
    #     while not done:
    #         action = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         score += reward
    #
    #         if not args.load_checkpoint:
    #             agent.store_transition(observation, action, reward,
    #                                    observation_, int(done))
    #             agent.learn()
    #         observation = observation_
    #         n_steps += 1
    #     scores.append(score)
    #     steps_array.append(n_steps)
    #
    #     avg_score = np.mean(scores[-100:])
    #     print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
    #           'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
    #     if avg_score > best_score:
    #         if not args.load_checkpoint:
    #             agent.save_models()
    #         best_score = avg_score
    #
    #     eps_history.append(agent.epsilon)
    #     if args.load_checkpoint and n_steps >= 18000:
    #         break

    # plot_learning(steps_array, scores, eps_history, figure_file)
    # np.save(scores_file, np.array(scores))
