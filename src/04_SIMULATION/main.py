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

    # # METHOD 1
    # # np.random.seed(0)
    env = FixedSimEnv()
    next_obs, rew, done, info = env.reset()
    print(env.lines[ROUTES[0]].hist_date)

    i = 0
    while not done and i < MAX_STEPS:
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
    next_obs, rew, done, info = env.reset(random_date=False, 
                                          hist_date=env.lines[ROUTES[0]].hist_date)
    print(env.lines[ROUTES[0]].hist_date)
    i = 0
    while not done and i < MAX_STEPS:
        next_obs, rew, done, info = env.step()

        if not done:
            control_vehs = flag_departure_event(info)

            if not control_vehs.empty:
                control_veh = env.vehicles[control_vehs.index[0]]
                rt_id = control_vehs['route_id'].iloc[0]

                if len(control_veh.next_trips) == 0:
                    print(control_vehs.iloc[0])
                    print(control_veh.past_trips)
                    print(control_veh.curr_trip)
                    print(control_veh.next_trips)

                schd_dep_t = control_veh.next_trips[0].schedule['departure_time_sec'].values[0]
                earliest_dep_t = max(schd_dep_t-MAX_EARLY_DEV*60, env.time)
                layover_buses = info[(info['route_id']==control_veh.route_id) & 
                                    (info['stop_id']==control_vehs['stop_id'].iloc[0]) & 
                                    (info['status'].isin([1,4])) & 
                                    (info['trip_sequence'] < control_vehs['trip_sequence'].iloc[0])].copy()
                layover_bus_dep = None
                if not layover_buses.empty:
                    layover_bus_dep = layover_buses['next_event_t'].max().total_seconds()
                    layover_seq = layover_buses['trip_sequence'].tolist()
                    curr_trip_seq = control_vehs['trip_sequence'].iloc[0]
                    print(f'layover trip sequence {layover_seq}')
                    print(f'current trip sequence {curr_trip_seq}')

                pre_hw, next_hw = control_veh.compute_headways(
                    env.lines[rt_id], earliest_dep_t, env.routes[rt_id], terminal=True,
                    layover_bus_dep=layover_bus_dep)
                
                if pre_hw is not None and next_hw is not None:
                    if pre_hw >= 0:
                        latest_dep_t = max(schd_dep_t+MAX_LATE_DEV*60, env.time)
                        rec_dep_t = recommended_dep_t(pre_hw, next_hw, env.time)
                        dep_t = min(latest_dep_t, rec_dep_t)
                        dep_t = max(earliest_dep_t, dep_t)

                        # earliest_td = pd.to_timedelta(round(earliest_dep_t), unit='S')
                        # time_td = pd.to_timedelta(round(env.time), unit='S')
                        # sch_td = pd.to_timedelta(round(schd_dep_t), unit='S')
                        # latest_td = pd.to_timedelta(round(latest_dep_t), unit='S')
                        # rec_td = pd.to_timedelta(round(rec_dep_t), unit='S')
                        # dep_td = pd.to_timedelta(round(dep_t), unit='S')
                        # pre_td = pd.to_timedelta(round(pre_hw), unit='S')
                        # post_td = pd.to_timedelta(round(next_hw), unit='S')

                        # print(f'time {time_td}')
                        # print(f'headways {pre_td} and {post_td}')
                        # print(f'scheduled departure {sch_td}')
                        # print(f'latest departure {latest_td}')
                        # print(f'earliest departure {earliest_td}')
                        # print(f'recommended departure {rec_td}')
                        # print(f'departure time {dep_td}')

                        if dep_t > control_veh.next_event['t']:
                            control_veh.next_event['t'] = deepcopy(dep_t)
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

