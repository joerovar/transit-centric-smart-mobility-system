import argparse

import pandas as pd

import agents_dqn as Agents
from utils import plot_learning
import sim_env
import post_process
from input import *
import os

if __name__ == '__main__':
    tstamp_save = time.strftime("%m%d-%H%M")
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
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-simple_reward', type=bool, default=False,
                        help='delayed(false)/simple(true)')
    parser.add_argument('-fc_dims', type=int, default=256,
                        help='fully connected dimensions')
    parser.add_argument('-weight_ride_time', type=float, default=0.0,
                        help='weight for ride time in reward')

    parser.add_argument('-env', type=str, default=tstamp_save,
                        help='environment')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-model_base_path', type=str, default='out/trained_nets/',
                        help='path for model saving/loading')
    parser.add_argument('-test_save_folder', type=str, default='RL',
                        help='DDQN-LA/DDQN-HA')
    args = parser.parse_args()

    if not args.load_checkpoint:
        parent_dir = 'out/trained_nets/'
        child_dir = args.env
        path = os.path.join(parent_dir, child_dir)
        os.mkdir(path)

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
                   chkpt_dir=args.model_base_path + args.env + '/',
                   algo=args.algo,
                   env_name=args.env,
                   fc_dims=args.fc_dims)
    if not args.load_checkpoint:

        # --------------------------------------------------- TRAINING -----------------------------------------
        arg_params = {'param': ['n_games', 'lr', 'eps_min', 'gamma', 'eps_dec', 'eps', 'max_mem', 'bs',
                                'replace', 'algo', 'simple_rew', 'fc_dims', 'weight_ride_time', 'limit_hold'],
                      'value': [args.n_games, args.lr, args.eps_min, args.gamma, args.gamma, args.eps, args.max_mem,
                                args.bs, args.replace, args.algo, args.simple_reward, args.fc_dims,
                                args.weight_ride_time, LIMIT_HOLDING]}

        figure_file = 'out/trained_nets/' + args.env + '/rew_curve.png'
        scores_file = 'out/trained_nets/' + args.env + '/rew_nums.csv'
        params_file = 'out/trained_nets/' + args.env + '/input_params.csv'

        df_params = pd.DataFrame(arg_params)
        df_params.to_csv(params_file, index=False)

        scores, eps_history, steps = [], [], []
        n_steps = 0
        for j in range(args.n_games):
            score = 0
            env = sim_env.DetailedSimulationEnvWithDeepRL()
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
                    if route_progress == 0.0 or pax_at_stop == 0 or previous_denied or args.simple_reward:
                        action = agent.choose_action(observation, mask_idx=0)
                    else:
                        action = agent.choose_action(observation)
                    env.take_action(action)
                env.update_rewards(simple_reward=args.simple_reward, weight_ride_time=args.weight_ride_time)
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
        for j in range(args.n_games):
            score = 0
            env = sim_env.DetailedSimulationEnvWithDeepRL()
            done = env.reset_simulation()
            done = env.prep()
            while not done:
                if not env.bool_terminal_state:
                    trip_id = env.bus.active_trip[0].trip_id
                    all_sars = env.trips_sars[trip_id]
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
                    if route_progress == 0.0 or pax_at_stop == 0 or previous_denied or args.simple_reward:
                        action = agent.choose_action(observation, mask_idx=0)
                    else:
                        action = agent.choose_action(observation)
                    env.take_action(action)
                done = env.prep()
            env.process_results()
            trajectories_set.append(env.trajectories)
            sars_set.append(env.trips_sars)
            pax_set.append(env.completed_pax)

        path_trajectories = 'out/' + args.save_folder + '/trajectory_set_' + tstamp + '.pkl'
        path_sars = 'out/' + args.save_folder + '/sars_set_' + tstamp + '.pkl'
        path_completed_pax = 'out/' + args.save_folder + '/pax_set_' + tstamp + '.pkl'
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_sars, sars_set)
        post_process.save(path_completed_pax, pax_set)
        #
        # trajectories_set = load('out/DDQN-LA/trajectory_set_0122-145334.pkl')
        # load_mean, _, _, ons_mean, _ = pax_per_trip_from_trajectory_set(trajectories_set, IDX_LOAD,
        #                                                                 IDX_PICK, IDX_DROP, STOPS)
        #
        # stable = (FOCUS_TRIPS_MEAN_HW, FOCUS_TRIPS_MEAN_HW, FOCUS_TRIPS_MEAN_HW)
        # early = (FOCUS_TRIPS_MEAN_HW-FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW+FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW)
        # late = (FOCUS_TRIPS_MEAN_HW+FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW-FOCUS_TRIPS_MEAN_HW/2, FOCUS_TRIPS_MEAN_HW)
        # policies = np.zeros((3, len(CONTROLLED_STOPS)-1, N_ACTIONS_RL-1))
        # route_progress_scenarios = np.array([(STOPS.index(s) / len(STOPS)) for s in CONTROLLED_STOPS[:-1]])
        # avg_load = np.array([load_mean[STOPS.index(s)] for s in CONTROLLED_STOPS[:-1]])
        # avg_ons = np.array([ons_mean[STOPS.index(s)] for s in CONTROLLED_STOPS[:-1]])
        # k = 0
        # for scenario in (stable, early, late):
        #     for i in range(route_progress_scenarios.size):
        #         obs = np.array([route_progress_scenarios[i], avg_load[i], scenario[0], scenario[1], avg_ons[i], scenario[2]], dtype=np.float32)
        #         q_values = agent.q_values(obs).detach().numpy()[0][1:]
        #         temp = np.argsort(q_values)
        #         rank_q_values = np.empty_like(temp)
        #         rank_q_values[temp] = np.arange(len(q_values))
        #         if scenario == stable and i == 0:
        #             x = rank_q_values[-1]
        #             rank_q_values[-1] = rank_q_values[-2]
        #             rank_q_values[-2] = x
        #         policies[k, i] = rank_q_values
        #     k += 1
        # fig_names = ['on time', 'early', 'late']
        # for i in range(policies.shape[0]):
        #     plt.imshow(policies[i].T, cmap='Greens')
        #     plt.xticks(np.arange(len(route_progress_scenarios)), np.arange(len(route_progress_scenarios)) + 1, fontsize=10)
        #     plt.yticks(np.arange(N_ACTIONS_RL-1), ['$T_{hold} = 0.0\omega H$', '$T_{hold} = 0.25\omega H$', '$T_{hold} = 0.50\omega H$', '$T_{hold} = 0.75\omega H$', '$T_{hold} = 1.0\omega H$'], fontsize=10)
        #     plt.xlabel('control stop', fontsize=10)
        #     plt.title(fig_names[i] + ' bus', fontsize=10)
        #     plt.savefig('out/'+ args.save_folder +'/policy_' + fig_names[i] + '.png')
        #     plt.close()

# cd src/04_SIMULATION
# train
# python main.py -algo -n_games -simple_reward
# test
# python main.py -env (tstamp) -algo -n_games -eps -load_checkpoint -test_save_folder  -simple_reward
