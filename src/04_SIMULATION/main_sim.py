import random
import numpy as np
import pandas as pd
from sim_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl, DetailedSimulationEnvWithDeepRL
from file_paths import *
import post_process
from datetime import datetime, timedelta
from output import dwell_times, error_headway, link_times
from output import PostProcessor
from input import FOCUS_TRIP_IDS_OUT_LONG, FOCUS_TRIP_IDS_OUT_SHORT, FOCUS_TRIP_DEP_T_OUT_LONG, FOCUS_TRIP_DEP_T_OUT_SHORT
st = time.time()


def run_base_detailed(episodes=3, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    all_delays = []
    all_trip_times = []
    for i in range(episodes):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            # trajectories_set.append(env.trajectories)
            # pax_set.append(env.completed_pax)
    # if save:
        # path_trajectories = 'out/NC/trajectories_set_' + tstamp + ext_var
        # path_completed_pax = 'out/NC/pax_set_' + tstamp + ext_var
        # post_process.save(path_trajectories, trajectories_set)
        # post_process.save(path_completed_pax, pax_set)
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


def run_sample_rl(episodes=1, simple_reward=False):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    sars_set = []
    pax_set = []
    pax_details = []
    for j in range(episodes):
        env = DetailedSimulationEnvWithDeepRL()
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            trip_id = env.bus.active_trip[0].trip_id
            all_sars = env.trips_sars[trip_id]
            if not env.bool_terminal_state:
                observation = np.array(all_sars[-1][0], dtype=np.float32)
                route_progress = observation[0]
                pax_at_stop = observation[4]
                curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
                previous_denied = False
                for p in curr_stop[0].pax.copy():
                    if p.arr_time <= env.time:
                        if p.denied:
                            previous_denied = True
                            break
                    else:
                        break
                if route_progress == 0.0 or pax_at_stop == 0 or previous_denied:
                    action = random.randint(1, 4)
                else:
                    action = random.randint(0, 4)
                env.take_action(action)
            env.update_rewards(simple_reward=simple_reward)
            done = env.prep()
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


run_base_detailed(save=True)
# analyze_delays()
# run_sample_rl(episodes=5, simple_reward=True)
# run_base_detailed(episodes=25, save=True)
# run_base_control_detailed(episodes=25, save=True)
# other tstamps

path_tr_nc = 'out/NC/trajectories_set_0127-184104.pkl'
path_p_nc = 'out/NC/pax_set_0127-184104.pkl'

path_tr_eh = 'out/EH/trajectories_set_0127-184112.pkl'
path_p_eh = 'out/EH/pax_set_0127-184112.pkl'

path_p_rl2 = 'out/DDQN-LA/pax_set_0127-184208.pkl'
path_tr_rl2 = 'out/DDQN-LA/trajectory_set_0127-184208.pkl'

path_tr_rl1 = 'out/DDQN-HA/trajectory_set_0127-190733.pkl'
path_p_rl1 = 'out/DDQN-HA/pax_set_0127-190733.pkl'

# #

# PROCESS RAW RESULTS
# path_trips = [path_tr_nc, path_tr_eh, path_tr_rl2, path_tr_rl1]
# path_pax = [path_p_nc, path_p_eh, path_p_rl2, path_p_rl1]
# tags = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA']
# post_processor = PostProcessor(path_trips, path_pax, tags)
# post_processor.write_trajectories()
# post_processor.total_trip_time_distribution()
# post_processor.headway()
# post_processor.load_profile()
# post_processor.denied()
# post_processor.hold_time()
# post_processor.wait_times_per_stop()
# post_processor.pax_times()


# VALIDATION
# path_trips = [path_tr_nc]
# path_pax = [path_p_nc]
# post_processor = PostProcessor(path_trips, path_pax, ['NC'])
# post_processor.departure_delay_validation()
# post_processor.dwell_time_validation()
# post_processor.trip_time_dist_validation()
# post_processor.load_profile_validation()
# post_processor.load_profile_base()

# PROCESS TRAJECTORIES FILE
# stops = post_process.load('in/xtr/rt_20-2019-09/route_stops.pkl')
# links = [(i, j) for i, j in zip(stops[:-1], stops[1:])]
# df_nc = pd.read_csv('out/trajectories0.csv')
# df_eh = pd.read_csv('out/trajectories1.csv')
# df_rl = pd.read_csv('out/trajectories2.csv')
# nr_replications = 25
# header = ['stop', 'mean1', 'std1', 'mean2', 'std2', 'mean3', 'std3']
# header_cv = ['stop', 'cv1', 'mean2', 'std2', 'mean3', 'std3']

# dwell_times()
# link_times()
# error_headway(stops, df_nc, df_eh, df_rl, nr_replications)

print("ran in %.2f seconds" % (time.time()-st))
