from sim_env import SimEnv, FixedSimEnv
from sim_env import recommended_dep_t, flag_departure_event
from data_handlers import write_sim_data
import pandas as pd
import os
from constants import *
from copy import deepcopy
import numpy as np

def save_df(df, pth_from_cwd, filename):
    df.to_csv(
        os.path.join(SRC_PATH, 
                     pth_from_cwd, filename), index=False
    )
    return

def td(t):
    return pd.to_timedelta(round(t), unit='S')

def get_layover_buses(info, rt, stop_id, trip_seq):
    return info[(info['route_id']==rt) & 
                (info['stop_id']==stop_id) & 
                (info['status'].isin([1,2,4])) & 
                (info['stop_sequence']==1) & 
                (info['trip_sequence'] < trip_seq)].copy()

def run_base(env, done):
    i = 0
    while not done and i < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        i += 1
    return env

def run_ehd(env, done):
    i = 0
    while not done and i < MAX_STEPS:
        next_obs, rew, done, info = env.step()

        if not done:
            control_vehs = flag_departure_event(info)

            if not control_vehs.empty:
                control_veh = env.vehicles[control_vehs.index[0]]
                rt_id = control_vehs['route_id'].iloc[0]
                stop_id = control_vehs['stop_id'].iloc[0]
                trip_seq = control_vehs['trip_sequence'].iloc[0]

                if len(control_veh.next_trips) == 0:
                    print(control_vehs.iloc[0])
                    print(control_veh.past_trips, control_veh.curr_trip, control_veh.next_trips)

                next_trip = control_veh.next_trips[0]
                schd_dep_t = next_trip.schedule['departure_time_sec'].values[0]
                earliest_dep_t = max(schd_dep_t-MAX_EARLY_DEV*60, env.time)

                layover_buses = get_layover_buses(info, rt_id,stop_id, trip_seq)
                
                layover_bus_dep = None
                if not layover_buses.empty:
                    layover_bus_dep = layover_buses['next_event_t'].max().total_seconds()

                pre_hw, next_hw = control_veh.compute_headways(
                    env.lines[rt_id], earliest_dep_t, 
                    env.routes[rt_id], terminal=True,
                    layover_bus_dep=layover_bus_dep)
                
                if pre_hw is not None and next_hw is not None and pre_hw >= 0:
                    latest_dep_t = max(schd_dep_t+MAX_LATE_DEV*60, env.time)
                    rec_dep_t = recommended_dep_t(pre_hw, next_hw, env.time)
                    dep_t = min(latest_dep_t, rec_dep_t)
                    dep_t = max(earliest_dep_t, dep_t)

                    # print(f'time {td(env.time)}')
                    # print(f'headways {td(pre_hw)} and {td(next_hw)}')
                    # print(f'scheduled {td(schd_dep_t)}')
                    # print(f'latest {td(latest_dep_t)} earliest {td(earliest_dep_t)}')
                    # print(f'recommended {td(rec_dep_t)} actual {td(dep_t)}')

                    if dep_t > control_veh.next_event['t']:
                        control_veh.next_event['t'] = deepcopy(dep_t)
                        new_df_info = env._get_info_vehicles()
                        new_df_info['nr_step'] = deepcopy(env.step_counter)
                        env.info_records[-1] = new_df_info.copy()
        i += 1
    return env
    

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
        env = run_base(env, done)
        # df_events = pd.concat(env.info_records, ignore_index=True) # for animations 
        df_pax = env.get_pax_records(scenario='NC') # for pax experience
        df_trips = env.get_trip_records(scenario='NC') # for trip experience
        lst_pax.append(df_pax)
        lst_trips.append(df_trips)

        # METHOD 2 DON'T RESET DATE
        # np.random.seed(0)
        next_obs, rew, done, info = env.reset(hist_date=day)
        print(env.hist_date)
        env = run_ehd(env, done)

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
