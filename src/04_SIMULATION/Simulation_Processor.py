from RL_Agents import DDQNAgent
from Output_Processor import plot_learning
from Variable_Inputs import *
import os
import Simulation_Envs
import time
from sb3_contrib.ppo_mask import MaskablePPO
from run import BusEnv
import numpy as np


def process_trip_record(record, record_col_names, rep_nr):
    df = pd.DataFrame(record, columns=record_col_names)
    df['replication'] = pd.Series([rep_nr + 1 for _ in range(len(df.index))])
    return df


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


def run_base_dispatching(replications, capacity, save_results=False, control_strategy=None, save_folder=None,
                         cancelled_blocks=None):
    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []
    for i in range(replications):
        env = Simulation_Envs.SimulationEnvWithCancellations(control_strategy=control_strategy,
                                                             cancelled_blocks=cancelled_blocks, bus_capacity=capacity)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
            if env.obs and not done:
                env.dispatch_decision()
        if save_results:
            out_trip_record_set.append(process_trip_record(env.out_trip_record, OUT_TRIP_RECORD_COLS, i))
            in_trip_record_set.append(process_trip_record(env.in_trip_record, IN_TRIP_RECORD_COLS, i))
            pax_record_set.append(process_trip_record(env.completed_pax_record, PAX_RECORD_COLS, i))

    if save_results:
        write_trip_records(save_folder, out_trip_record_set, in_trip_record_set, pax_record_set)
    return


def rl_dispatch(n_episodes, train=False, prob_cancel=0.0, weight_hold_t=0.0, save_results=False, save_folder=None,
                cancelled_blocks=None, tstamp_policy=None, default_action=None):
    if not tstamp_policy:
        assert train
        tstamp_policy = time.strftime("%m%d-%H%M")
    # agent_ = getattr(Agents, ALGO)
    path_train_info = DIR_ROUTE_OUTS + 'trained_nets/' + tstamp_policy + '_dispatch'
    agent = DDQNAgent(gamma=DISCOUNT_FACTOR, epsilon=EPS, lr=LEARN_RATE, input_dims=[NR_STATE_D_RL],
                      n_actions=NR_ACTIONS_D_RL, mem_size=MAX_MEM, eps_min=EPS_MIN, batch_size=BATCH_SIZE,
                      replace=EPOCHS_REPLACE, eps_dec=EPS_DEC, chkpt_dir=path_train_info + '/',
                      algo=ALGO, fc_dims=FC_DIMS)

    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []
    best_score = -np.inf
    if train:
        os.mkdir(path_train_info)

        arg_params = {'param': ['n_episodes', 'lr', 'eps_min', 'gamma', 'eps_dec', 'eps', 'max_mem', 'bs',
                                'replace', 'algo', 'fc_dims'],
                      'value': [n_episodes, LEARN_RATE, EPS_MIN, DISCOUNT_FACTOR, EPS_DEC, EPS, MAX_MEM,
                                BATCH_SIZE, EPOCHS_REPLACE, ALGO, FC_DIMS]}
        df_params = pd.DataFrame(arg_params)
        params_file = path_train_info + '/input_params.csv'
        df_params.to_csv(params_file, index=False)
    else:
        agent.load_models()
    figure_file = path_train_info + '/rew_curve.png'
    scores_file = path_train_info + '/rew_nums.csv'
    scores, eps_history, steps = [], [], []
    n_steps = 0

    env = Simulation_Envs.SimulationEnvWithCancellations(prob_cancelled_block=prob_cancel,
                                                         control_strategy='RL', weight_hold_t=weight_hold_t)
    for j in range(n_episodes):
        score = 0
        done = env.reset_simulation()
        while not done:
            done = env.prep()
            if env.obs and not done:
                if env.prev_reward and train:
                    obs = np.array(env.prev_obs, dtype=np.float32)
                    obs_ = np.array(env.obs, dtype=np.float32)
                    score += env.prev_reward
                    prev_action = int(env.prev_hold_t / HOLD_INTERVALS)
                    agent.store_transition(obs, prev_action, env.prev_reward, obs_, 0)
                    agent.learn()
                    n_steps += 1
                sched_dev = env.obs[-1]
                obs_ = np.array(env.obs, dtype=np.float32)
                hold_t_max = max(IMPOSED_DELAY_LIMIT - sched_dev, 0)
                max_action = int(hold_t_max / HOLD_INTERVALS)
                if default_action:
                    action = default_action
                else:
                    if max_action == 0:
                        action = 0
                    elif max_action < NR_ACTIONS_D_RL - 1:
                        action = agent.choose_action(obs_, mask_idx=[i for i in range(max_action + 1, NR_ACTIONS_D_RL)])
                    else:
                        action = agent.choose_action(obs_)
                past_sched_hw = env.obs[PAST_HW_HORIZON + FUTURE_HW_HORIZON:PAST_HW_HORIZON * 2 + FUTURE_HW_HORIZON]
                past_actual_hw = env.obs[:PAST_HW_HORIZON]
                future_sched_hw = env.obs[
                                  PAST_HW_HORIZON * 2 + FUTURE_HW_HORIZON:PAST_HW_HORIZON * 2 + FUTURE_HW_HORIZON * 2]
                future_actual_hw = env.obs[PAST_HW_HORIZON:PAST_HW_HORIZON + FUTURE_HW_HORIZON]
                sched_dev = env.obs[-1]
                # print(f'current time is {str(timedelta(seconds=round(env.time)))} '
                #       f'and next event time is {str(timedelta(seconds=round(env.bus.next_event_time)))}')
                # print(f'trip {env.bus.pending_trips[0].trip_id}')
                # print(f'schedule deviation {round(env.obs[-1])}')
                # print(f'sched hw {[str(timedelta(seconds=round(hw))) for hw ins past_sched_hw]} | {[str(timedelta(seconds=round(hw))) for hw ins future_sched_hw]}')
                # print(f'actual hw {[str(timedelta(seconds=round(hw))) for hw ins past_actual_hw]} | {[str(timedelta(seconds=round(hw))) for hw ins future_actual_hw]}')
                # print(f'holding time {action*HOLD_INTERVALS}')
                env.dispatch_decision(hold_time=action * HOLD_INTERVALS)
        if not train and save_results:
            out_trip_record_set.append(process_trip_record(env.out_trip_record, OUT_TRIP_RECORD_COLS, j))
            in_trip_record_set.append(process_trip_record(env.in_trip_record, IN_TRIP_RECORD_COLS, j))
            pax_record_set.append(process_trip_record(env.completed_pax_record, PAX_RECORD_COLS, j))
        if train:
            scores.append(score)
            steps.append(n_steps)
            avg_score = np.mean(scores[-100:])
            print('episode ', j, 'score %.2f' % score, 'average score %.2f' % avg_score,
                  'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
            if avg_score > best_score:
                agent.save_models()
                best_score = avg_score
            eps_history.append(agent.epsilon)
    if train:
        plot_learning(steps, scores, figure_file, epsilons=eps_history)
        scores_d = {'step': steps, 'score': scores, 'eps': eps_history}
        scores_df = pd.DataFrame(scores_d)
        scores_df.to_csv(scores_file, index=False)
    if not train and save_results:
        write_trip_records(save_folder, tstamp, out_trip_record_set, in_trip_record_set, pax_record_set)
        key_params = {'prob_cancel': [prob_cancel], 'n_replications': [n_episodes],
                      'cancelled_blocks': [cancelled_blocks]}
        pd.DataFrame(key_params).to_csv(DIR_ROUTE_OUTS + save_folder + '/' + tstamp + '-key_params.csv', index=False)
    return


def run_base(replications=4, save_results=False, control_eh=False, hold_adj_factor=0.0, tt_factor=1.0,
             control_strength=0.7):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []
    for i in range(replications):
        if control_eh:
            env = Simulation_Envs.SimulationEnvWithHolding(hold_adj_factor=hold_adj_factor, tt_factor=tt_factor,
                                                           control_strength=control_strength)
        else:
            env = Simulation_Envs.SimulationEnv(tt_factor=tt_factor)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save_results:
            out_trip_record_set.append(process_trip_record(env.out_trip_record, OUT_TRIP_RECORD_COLS, i))
            in_trip_record_set.append(process_trip_record(env.in_trip_record, IN_TRIP_RECORD_COLS, i))
            pax_record_set.append(process_trip_record(env.completed_pax_record, PAX_RECORD_COLS, i))

    if save_results:
        scenario = 'EH' if control_eh else 'NC'
        write_trip_records(scenario, tstamp, out_trip_record_set, in_trip_record_set, pax_record_set)
        if control_eh:
            params = {'param': ['control_strength'], 'value': [control_strength]}
            df_params = pd.DataFrame(params)
            df_params.to_csv(DIR_ROUTE_OUTS + 'EH/' + tstamp + '-params_used.pkl', index=False)
    return


def train_rl(n_episodes_train, simple_reward=False):
    tstamp_policy = time.strftime("%m%d-%H%M")
    # agent_ = getattr(Agents, ALGO)
    path_train_info = DIR_ROUTE_OUTS + 'trained_nets/' + tstamp_policy + '_at_stop'
    agent = DDQNAgent(gamma=DISCOUNT_FACTOR, epsilon=EPS, lr=LEARN_RATE,
                      input_dims=[N_STATE_PARAMS_RL], n_actions=N_ACTIONS_RL,
                      mem_size=MAX_MEM, eps_min=EPS_MIN, batch_size=BATCH_SIZE, replace=EPOCHS_REPLACE, eps_dec=EPS_DEC,
                      chkpt_dir=path_train_info + '/', algo=ALGO, fc_dims=FC_DIMS)
    best_score = -np.inf
    os.mkdir(path_train_info)
    figure_file = path_train_info + '/rew_curve.png'
    scores_file = path_train_info + '/rew_nums.csv'
    params_file = path_train_info + '/input_params.csv'

    arg_params = {'param': ['n_episodes', 'lr', 'eps_min', 'gamma', 'eps_dec', 'eps', 'max_mem', 'bs',
                            'replace', 'algo', 'simple_rew', 'fc_dims', 'weight_ride_time', 'limit_hold',
                            'tt_factor', 'hold_adj_factor', 'estimated_pax'],
                  'value': [n_episodes_train, LEARN_RATE, EPS_MIN, DISCOUNT_FACTOR, EPS_DEC, EPS, MAX_MEM,
                            BATCH_SIZE, EPOCHS_REPLACE, ALGO, simple_reward, FC_DIMS,
                            WEIGHT_RIDE_T, LIMIT_HOLDING, TT_FACTOR, HOLD_ADJ_FACTOR, ESTIMATED_PAX]}
    df_params = pd.DataFrame(arg_params)
    df_params.to_csv(params_file, index=False)

    scores, eps_history, steps = [], [], []
    n_steps = 0
    for j in range(n_episodes_train):
        score = 0
        env = Simulation_Envs.SimulationEnvWithRL(hold_adj_factor=HOLD_ADJ_FACTOR,
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
    plot_learning(steps, scores, figure_file, epsilons=eps_history)
    scores_d = {'step': steps, 'score': scores, 'eps': eps_history}
    scores_df = pd.DataFrame(scores_d)
    scores_df.to_csv(scores_file, index=False)
    return


def test_rl(n_episodes_test, tstamp_policy, save_results=False, simple_reward=False):
    # agent_ = getattr(Agents, ALGO)
    path_train_info = DIR_ROUTE_OUTS + 'trained_nets/' + tstamp_policy + '_at_stop'
    agent = DDQNAgent(gamma=DISCOUNT_FACTOR, epsilon=EPS, lr=LEARN_RATE, input_dims=[N_STATE_PARAMS_RL],
                      n_actions=N_ACTIONS_RL, mem_size=MAX_MEM, eps_min=EPS_MIN, batch_size=BATCH_SIZE,
                      replace=EPOCHS_REPLACE, eps_dec=EPS_DEC, chkpt_dir=path_train_info + '/',
                      algo=ALGO, fc_dims=FC_DIMS)
    agent.load_models()
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    out_trip_record_set = []
    in_trip_record_set = []
    pax_record_set = []
    for j in range(n_episodes_test):
        env = Simulation_Envs.SimulationEnvWithRL(tt_factor=TT_FACTOR, hold_adj_factor=HOLD_ADJ_FACTOR,
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
            out_trip_record_set.append(process_trip_record(env.out_trip_record, OUT_TRIP_RECORD_COLS, j))
            in_trip_record_set.append(process_trip_record(env.in_trip_record, IN_TRIP_RECORD_COLS, j))
            pax_record_set.append(process_trip_record(env.completed_pax_record, PAX_RECORD_COLS, j))
    if save_results:
        scenario = 'DDQN-HA'
        write_trip_records(scenario, tstamp, out_trip_record_set, in_trip_record_set, pax_record_set)

        with open(DIR_ROUTE_OUTS + scenario + '/' + tstamp + '-net_used.csv', 'w') as f:
            f.write(str(tstamp_policy))
    return
