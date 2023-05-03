from sim_env import SimEnv, FixedSimEnv
from sim_env import recommended_dep_t, flag_departure_event
from data_handlers import write_sim_data
import pandas as pd
import os
from constants import *
from copy import deepcopy
import numpy as np

if __name__ == '__main__':
    # if not written yet
    # write_sim_data()

    
    tstamp = pd.Timestamp.today().strftime('%m%d-%H%M')
    path_from_cwd = 'data/sim_out/experiments_' + tstamp
    os.mkdir(os.path.join(SRC_PATH, path_from_cwd))

    # METHOD 1
    # np.random.seed(0)
    env = FixedSimEnv()
    next_obs, rew, done, info = env.reset()
    print(env.line.hist_date)

    i = 0
    while not done and i < 10000:
        next_obs, rew, done, info = env.step()
        i += 1

    os.mkdir(os.path.join(SRC_PATH, path_from_cwd + '/NC'))
    # for animations
    df_events = pd.concat(env.info_records, ignore_index=True)
    df_events.to_csv(os.path.join(SRC_PATH, 
                                path_from_cwd + '/NC', 
                                'events.csv'), index=False)

    # for pax experience
    df_pax = env.demand.pax_served.copy()
    df_pax.to_csv(
        os.path.join(SRC_PATH, 
                     path_from_cwd + '/NC', 
                     'pax.csv'), index=False
    )

    # for trip experience
    trip_records = []
    for veh in env.vehicles:
        if not veh.trip_records.empty:
            trip_records.append(veh.trip_records)
    df_trips = pd.concat(trip_records, ignore_index=True)

    df_trips.to_csv(
        os.path.join(SRC_PATH,
                     path_from_cwd + '/NC',
                     'trips.csv'), index=False
    )


    # METHOD 2 DON'T RESET DATE
    # np.random.seed(0)
    next_obs, rew, done, info = env.reset(reset_date=False)
    print(env.line.hist_date)

    while not done and i < 10000:
        next_obs, rew, done, info = env.step()

        control_veh = flag_departure_event(info, 'East')

        if not control_veh.empty:
            schd_dep_t = env.vehicles[control_veh.index[0]].next_trips[0].schedule['departure_time_sec'].values[0]
            earliest_dep_t = max(schd_dep_t-MAX_EARLY_DEV*60, env.time)
            pre_hw, next_hw = env.vehicles[control_veh.index[0]].compute_headways(env.line, earliest_dep_t, env.route, terminal=True)
            if pre_hw is not None and next_hw is not None:
                latest_dep_t = max(schd_dep_t+MAX_LATE_DEV*60, env.time)
                dep_t = min(latest_dep_t, recommended_dep_t(pre_hw, next_hw, env.time))

                env.vehicles[control_veh.index[0]].next_event['t'] = deepcopy(dep_t)
                new_df_info = env._get_info_vehicles()
                new_df_info['nr_step'] = deepcopy(env.step_counter)
                env.info_records[-1] = new_df_info.copy()

        i += 1

    os.mkdir(os.path.join(SRC_PATH, path_from_cwd + '/EHD'))
    # for animations
    df_events = pd.concat(env.info_records, ignore_index=True)
    df_events.to_csv(os.path.join(SRC_PATH, 
                                path_from_cwd + '/EHD', 
                                'events.csv'), index=False)

    # for pax experience
    df_pax = env.demand.pax_served.copy()
    df_pax.to_csv(
        os.path.join(SRC_PATH, 
                     path_from_cwd + '/EHD', 
                     'pax.csv'), index=False
    )

    # for trip experience
    trip_records = []
    for veh in env.vehicles:
        if not veh.trip_records.empty:
            trip_records.append(veh.trip_records)
    df_trips = pd.concat(trip_records, ignore_index=True)

    df_trips.to_csv(
        os.path.join(SRC_PATH,
                     path_from_cwd + '/EHD',
                     'trips.csv'), index=False
    )

