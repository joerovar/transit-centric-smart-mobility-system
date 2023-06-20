from sim_env import FixedSimEnv
from sim_env import recommended_dep_t, flag_departure_event
from data_handlers import write_sim_data
import pandas as pd
import os
from constants import *
from copy import deepcopy
import numpy as np

def ehd_message(tim, pre_hw, nxt_hw, schd_dep, 
                dtmax, dtmin, rec, act, exp_dep):
    print(f'----')
    print(f'Time {td(tim)}')
    print(f'Expected: {td(exp_dep)}')
    print(f'Headways: {td(pre_hw)} -- {td(nxt_hw)}')
    print(f'Schedule: {td(schd_dep)}')
    print(f'{td(dtmin)} < Departure < {td(dtmax)}')
    print(f'Recommended {td(rec)} ---> {td(act)}')
    return

def save_df(df, pth_from_cwd, filename):
    df.to_csv(
        os.path.join(SRC_PATH, 
                     pth_from_cwd, filename), index=False
    )
    return

def td(t):
    return pd.to_timedelta(round(t), unit='S')

def run_base(env, done, n_steps=0):
    while not done and n_steps < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        n_steps += 1
    return env, n_steps

def run_ehd(env, done, n_steps=0, debug=False):
    while not done and n_steps < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        n_steps += 1
        if done:
            continue

        control_vehs = flag_departure_event(info[1])

        if control_vehs.empty:
            continue

        control_veh = env.vehicles[control_vehs.index[0]]
        rt_id, trip_id = control_veh.route_id, control_veh.curr_trip.id
        min_dep_t, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
        # new min dep t refers to the potential imposed minimum
        # based on a layover bus that may be late
        pre_hw, new_min_dep_t = env.get_headway(info[1], control_vehs,
                                               min_dep_t, 
                                               terminal=True)
        if pre_hw is None:
            continue

        if pre_hw < 0:
            continue

        next_hw = env.get_next_headway(control_vehs, terminal=True,
                                       expected_dep=new_min_dep_t)       

        rec_dep_t = recommended_dep_t(pre_hw, next_hw, new_min_dep_t)
        new_dep_t = min(max_dep_t, rec_dep_t)

        if new_dep_t <= control_veh.next_event['t']:
            continue

        updated_info = env.adjust_departure(control_vehs, new_dep_t)

        if debug:
            next_trip = control_veh.next_trips[0]
            schd_dep_t = next_trip.schedule['departure_sec'].values[0]
            ehd_message(env.time, pre_hw, next_hw,
                        schd_dep_t, max_dep_t, min_dep_t, 
                        rec_dep_t, new_dep_t, new_min_dep_t)
            return env, info, n_steps, updated_info
    return env, n_steps
    

if __name__ == '__main__':
    # if not written yet
    # write_sim_data()
    
    tstamp = pd.Timestamp.today().strftime('%m%d-%H%M')
    path_from_cwd = 'data/sim_out/experiments_' + tstamp
    os.mkdir(os.path.join(SRC_PATH, path_from_cwd))

    env = FixedSimEnv()
    unique_dates = env.link_times['date'].unique()
    dates = np.random.choice(unique_dates, replace=False, size=3)

    lst_pax = []
    lst_trips = []

    for day in dates:
        # # METHOD 1
        # # np.random.seed(0)
        next_obs, rew, done, info = env.reset(hist_date=day)
        print(env.hist_date)
        env, n_steps = run_base(env, done, n_steps=0)
        # df_events = pd.concat(env.info_records, ignore_index=True) # for animations 
        df_pax = env.get_pax_records(scenario='NC') # for pax experience
        df_trips = env.get_trip_records(scenario='NC') # for trip experience
        lst_pax.append(df_pax)
        lst_trips.append(df_trips)

        # METHOD 2 DON'T RESET DATE
        # np.random.seed(0)
        next_obs, rew, done, info = env.reset(hist_date=day)
        print(env.hist_date)
        env, n_steps = run_ehd(env, done, n_steps=0)

        # df_events = pd.concat(env.info_records, ignore_index=True) # for animations 
        df_pax = env.get_pax_records(scenario='EHD') # for pax experience
        df_trips = env.get_trip_records(scenario='EHD') # for trip experience
        lst_pax.append(df_pax)
        lst_trips.append(df_trips)

    # save_df(df_events, scenario, 'events.csv')
    df_pax = pd.concat(lst_pax, ignore_index=True)
    df_trips = pd.concat(lst_trips, ignore_index=True)
    save_df(df_pax, path_from_cwd, 'pax.csv')
    save_df(df_trips, path_from_cwd, 'trips.csv')
