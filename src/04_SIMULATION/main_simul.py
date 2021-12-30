import random

import pandas as pd

from simulation_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl, DetailedSimulationEnvWithDeepRL
from file_paths import *
import post_process
from datetime import datetime, timedelta
from output import PostProcessor
st = time.time()


def run_base_detailed(episodes=2, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    # pax_details = []
    for i in range(episodes):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        # for s in env.stops:
        #     pax = s.pax
        #     pax_details += [(s.stop_id, str(timedelta(seconds=round(p.arr_time)))) for p in pax]
        # df_pax = pd.DataFrame(pax_details, columns=['orig_pax', 'arr_time'])
        # df_pax.to_csv('visualize_pax.csv', index=False)
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save:
        path_trajectories = 'out/NC/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/NC/pax_set_' + tstamp + ext_var
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_completed_pax, pax_set)
    return


def run_base_control_detailed(episodes=2, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for i in range(episodes):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save:
        path_trajectories = 'out/EH/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/EH/pax_set_' + tstamp + ext_var
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_completed_pax, pax_set)
    return


def run_sample_rl():
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    sars_set = []
    pax_set = []
    pax_details = []
    for j in range(1):
        env = DetailedSimulationEnvWithDeepRL()
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            if not env.bool_terminal_state:
                action = random.randint(0, 1)
                # action = random.randint(0, 4)
                env.take_action(action)
            env.update_rewards()
            done = env.prep()
        # env.process_results()
        # trajectories_set.append(env.trajectories)
        # sars_set.append(env.trips_sars)
        # pax_set.append(env.completed_pax)
        for sars in env.pool_sars:
            print(sars)
        pax_details += [(p.orig_idx, str(timedelta(seconds=round(p.arr_time))), str(timedelta(seconds=round(p.board_time))),
                         str(timedelta(seconds=round(p.wait_time))),
                         p.trip_id, p.denied) for p in env.completed_pax]
        df_pax = pd.DataFrame(pax_details, columns=['orig_pax', 'arr_time', 'board_time', 'wait_time', 'trip_id', 'denied'])
        df_pax.to_csv('visualize_pax.csv', index=False)

    # print(list(trajectories_set[0].keys()))
    return


def analyze_delays():
    nr_replications = 4
    replication_departures = []
    replication_arrivals = []
    for i in range(nr_replications):
        env = DetailedSimulationEnvWithControl()
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        replication_arrivals.append(env.log.recorded_arrivals)
        replication_departures.append(env.log.recorded_departures)
    dates = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']
    outbound_arr_stops = [386]
    inbound_arr_stops = [8613, 449]
    df_blocks = pd.read_csv('visual_trips.csv')
    df_stop_times = pd.read_csv('in/raw/route20_stop_time_merged.csv')
    trip_info_by_block = []
    unique_blocks = df_blocks['block_id'].unique().tolist()
    for block_id in unique_blocks:
        unique_trip_ids = df_blocks[df_blocks['block_id'] == block_id]['trip_id'].unique().tolist()
        for trip_id in unique_trip_ids:
            direction = df_blocks[df_blocks['trip_id'] == trip_id]['direction'].tolist()[0]
            avl_departures = []
            avl_arrivals = []
            trip_df = df_stop_times[df_stop_times['trip_id'] == trip_id]
            schd_dep_sec = df_blocks[df_blocks['trip_id'] == trip_id]['schd_sec'].tolist()[0]
            schd_dep_time = str(timedelta(seconds=round(schd_dep_sec)))
            schd_arr_sec = trip_df.sort_values(by='schd_sec')['schd_sec'].unique().tolist()[-1]
            schd_arr_time = str(timedelta(seconds=round(schd_arr_sec)))
            for d in dates:
                date_df = trip_df[trip_df['avl_arr_time'].astype(str).str[:10] == d]
                if direction == 'in':
                    arr_stops = inbound_arr_stops
                else:
                    arr_stops = outbound_arr_stops
                dep_df = date_df[date_df['stop_sequence'] == 1]
                if not dep_df.empty:
                    avl_dep_sec = dep_df['avl_dep_sec'].tolist()[0]
                    avl_departures.append(str(timedelta(seconds=round(avl_dep_sec % 86400))))
                else:
                    avl_departures.append('')
                arr_df = date_df[date_df['stop_id'].isin(arr_stops)]
                if not arr_df.empty:
                    avl_arr_sec = arr_df['avl_sec'].tolist()[0]
                    avl_arrivals.append(str(timedelta(seconds=round(avl_arr_sec%86400))))
                else:
                    avl_arrivals.append('')
            simul_departures = []
            simul_arrivals = []
            for replication in replication_departures:
                simul_dep_sec = replication[trip_id]
                if simul_dep_sec is not None:
                    simul_departures.append(str(timedelta(seconds=round(simul_dep_sec))))
                else:
                    simul_departures.append('')
            for replication in replication_arrivals:
                simul_arr_sec = replication[trip_id]
                if simul_arr_sec is not None:
                    simul_arrivals.append(str(timedelta(seconds=round(simul_arr_sec))))
                else:
                    simul_arrivals.append('')
            trip_info_by_block.append((block_id, trip_id, direction, schd_dep_sec, schd_dep_time, *avl_departures, *simul_departures))
            trip_info_by_block.append((block_id, trip_id, direction, schd_arr_sec, schd_arr_time, *avl_arrivals, *simul_arrivals))
    header_days = ['day' + str(i) for i in range(1, len(dates) + 1)]
    header_simul_reps = ['rep' + str(i) for i in range(1, nr_replications + 1)]
    df = pd.DataFrame(trip_info_by_block, columns=['block_id', 'trip_id', 'direction', 'schd_sec', 'schd_time', *header_days, *header_simul_reps])
    df.to_csv('delays.csv', index=False)
    return


# analyze_delays()
# run_sample_rl()
# run_base_detailed(episodes=10, save=True)
# run_base_control_detailed(episodes=10, save=True)
# other tstamps

path_tr_nc = 'out/NC/trajectories_set_1230-000823.pkl'
path_p_nc = 'out/NC/pax_set_1230-000823.pkl'
path_tr_eh = 'out/EH/trajectories_set_1230-000827.pkl'
path_p_eh = 'out/EH/pax_set_1230-000827.pkl'
path_tr_rl = 'out/RL/trajectory_set_1230-040209.pkl'
path_p_rl = 'out/RL/pax_set_1230-040209.pkl'
#
#
path_trips = [path_tr_nc, path_tr_eh, path_tr_rl]
path_pax = [path_p_nc, path_p_eh, path_p_eh]
tags = ['NC', 'EH', 'RL']
post_processor = PostProcessor(path_trips, path_pax, tags)
post_processor.write_trajectories()
post_processor.total_trip_time_distribution()
post_processor.headway()
post_processor.load_profile()
post_processor.wait_times()
post_processor.denied()
post_processor.hold_time()

# post_processor.load_profile_base()

#
# path_trips = [path_tr_nc]
# path_pax = [path_p_nc]
# post_processor = PostProcessor(path_trips, path_pax, ['NC'])
# post_processor.departure_delay_validation()
# post_processor.dwell_time_validation()
# post_processor.trip_time_dist_validation()
# post_processor.load_profile_validation()
print("ran in %.2f seconds" % (time.time()-st))
