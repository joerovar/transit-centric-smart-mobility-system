import random
import numpy as np
import pandas as pd
from sim_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl, DetailedSimulationEnvWithDeepRL
from file_paths import *
import post_process
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from output import dwell_times, error_headway, link_times
from output import PostProcessor
from input import FOCUS_TRIP_IDS_OUT_LONG, FOCUS_TRIP_IDS_OUT_SHORT, FOCUS_TRIP_DEP_T_OUT_LONG, FOCUS_TRIP_DEP_T_OUT_SHORT, STOPS
st = time.time()


def run_base_detailed(replications=4, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    all_delays = [[], []]
    all_trip_times = [[], []]
    for i in range(replications):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
            # for j in range(len(FOCUS_TRIP_IDS_OUT_LONG)):
            #     sched_dep = FOCUS_TRIP_DEP_T_OUT_LONG[j]
            #     actual_dep = env.log.recorded_departures[FOCUS_TRIP_IDS_OUT_LONG[j]]
            #     actual_arr = env.log.recorded_arrivals[FOCUS_TRIP_IDS_OUT_LONG[j]]
            #     all_delays[0].append(actual_dep-sched_dep)
            #     all_trip_times[0].append(actual_arr-actual_dep)
            # for j in range(len(FOCUS_TRIP_IDS_OUT_SHORT)):
            #     sched_dep = FOCUS_TRIP_DEP_T_OUT_SHORT[j]
            #     actual_dep = env.log.recorded_departures[FOCUS_TRIP_IDS_OUT_SHORT[j]]
            #     actual_arr = env.log.recorded_arrivals[FOCUS_TRIP_IDS_OUT_SHORT[j]]
            #     all_delays[1].append(actual_dep-sched_dep)
            #     all_trip_times[1].append(actual_arr-actual_dep)
    # print(all_delays)
    # post_process.save('in/actual_delays_out.pkl', all_delays)
    # post_process.save('in/actual_trip_times_out.pkl', all_trip_times)

    if save:
        path_trajectories = 'out/NC/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/NC/pax_set_' + tstamp + ext_var
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_completed_pax, pax_set)
    return


def run_base_control_detailed(replications=2, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for i in range(replications):
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
        env = DetailedSimulationEnvWithDeepRL(estimate_pax=True)
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
            env.update_rewards(simple_reward=simple_reward, weight_ride_time=0.3)
            done = env.prep()
    return


# BENCHMARK
# run_sample_rl(simple_reward=True, episodes=1)
# run_base_detailed(replications=40, save=True)
# run_base_control_detailed(replications=40, save=True)

path_tr_nc = 'out/NC/trajectories_set_0222-211726.pkl'
path_p_nc = 'out/NC/pax_set_0222-211726.pkl'
path_tr_eh = 'out/EH/trajectories_set_0222-234859.pkl'
path_p_eh = 'out/EH/pax_set_0222-234859.pkl'
path_tr_eh2 = 'out/EH/trajectories_set_0222-212202.pkl' # EH(0.7)
path_p_eh2 = 'out/EH/pax_set_0222-212202.pkl' # EH (0.7)
path_tr_rl0 = 'out/DDQN-LA/trajectory_set_0224-124312.pkl' # 0224-1234
path_p_rl0 = 'out/DDQN-LA/pax_set_0224-124312.pkl' # 0224-1234
path_tr_rl1 = 'out/DDQN-HA/trajectory_set_0222-233419.pkl' # 0222-2247
path_p_rl1 = 'out/DDQN-HA/pax_set_0222-233419.pkl'
path_tr_rl2 = 'out/DDQN-HA/trajectory_set_0223-183027.pkl' # 0223-1249
path_p_rl2 = 'out/DDQN-HA/pax_set_0223-183027.pkl'
path_tr_rl3 = 'out/DDQN-HA/trajectory_set_0222-233609.pkl' # 0222-2315
path_p_rl3 = 'out/DDQN-HA/pax_set_0222-233609.pkl'
path_tr_rl4 = 'out/DDQN-HA/trajectory_set_0223-220817.pkl' # 0223-2159
path_p_rl4 = 'out/DDQN-HA/pax_set_0223-220817.pkl'
# tags = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA(-)', 'DDQN-HA(3.0)', 'DDQN-HA(1.5)', 'DDQN-HA(1.0)']
# prc = PostProcessor([path_tr_nc, path_tr_eh, path_tr_rl0, path_tr_rl1, path_tr_rl2, path_tr_rl3, path_tr_rl4],
#                     [path_p_nc, path_p_eh, path_p_rl0, path_p_rl1, path_p_rl2, path_p_rl3, path_p_rl4], tags)
tags = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA']
prc = PostProcessor([path_tr_nc, path_tr_eh, path_tr_rl0, path_tr_rl2],
                    [path_p_nc, path_p_eh, path_p_rl0, path_p_rl2], tags)
path_dir = 'out/compare/benchmark/'
results = {}
results.update(prc.pax_times_fast())
results.update(prc.headway())
results_df = pd.DataFrame(results, columns=list(results.keys()))
results_df.to_csv('out/compare/benchmark/numer_results.csv', index=False)
rbt_od_set = post_process.load('out/compare/benchmark/rbt_od_benchmark.pkl')
rbt_od_set = [rbt_od_set[0]] + [rbt_od_set[1]] + [rbt_od_set[2]] + [rbt_od_set[4]]
for i in range(len(rbt_od_set)):
    rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
plt.boxplot(rbt_od_set, labels=tags, sym='')
plt.xticks(rotation=45)
plt.xlabel('method')
plt.ylabel('reliability buffer time (min)')
plt.legend()
plt.tight_layout()
plt.savefig(path_dir + 'rbt.png')
plt.close()

# prc.write_trajectories()
# df_nc = pd.read_csv('out/NC/trajectories.csv')
# error_headway(STOPS, df_nc, 40)

# SENSITIVITY ANALYSIS

# VARIABILITY RUN TIMES
# path_tr_rl0 = 'out/DDQN-LA/trajectory_set_0224-124312.pkl'
# path_p_rl0 = 'out/DDQN-LA/pax_set_0224-124312.pkl'
# path_tr_rl0_inc = 'out/DDQN-LA/trajectory_set_0225-153557.pkl'
# path_p_rl0_inc = 'out/DDQN-LA/pax_set_0225-153557.pkl'
# path_tr_rl0_dec = 'out/DDQN-LA/trajectory_set_0225-153624.pkl'
# path_p_rl0_dec = 'out/DDQN-LA/pax_set_0225-153624.pkl'
# path_tr_rl2 = 'out/DDQN-HA/trajectory_set_0223-183027.pkl'
# path_p_rl2 = 'out/DDQN-HA/pax_set_0223-183027.pkl'
# path_tr_rl2_inc = 'out/DDQN-HA/trajectory_set_0225-145011.pkl'
# path_p_rl2_inc = 'out/DDQN-HA/pax_set_0225-145011.pkl'
# path_tr_rl2_dec = 'out/DDQN-HA/trajectory_set_0225-145143.pkl'
# path_p_rl2_dec = 'out/DDQN-HA/pax_set_0225-145143.pkl'
# tags = ['DDQN-LA(low)', 'DDQN-HA(low)', 'DDQN-LA(medium)', 'DDQN-HA(medium)', 'DDQN-LA(high)', 'DDQN-HA(high)']
# prc = PostProcessor([path_tr_rl0_dec, path_tr_rl2_dec, path_tr_rl0, path_tr_rl2, path_tr_rl0_inc, path_tr_rl2_inc],
#                     [path_p_rl0_dec, path_p_rl2_dec, path_p_rl0, path_p_rl2, path_p_rl0_inc, path_p_rl2_inc], tags)
# path_folder = 'out/compare/sensitivity run times/'
# results = {}
# results.update(prc.pax_times_fast(path_dir=path_folder))
# results.update(prc.headway(path_dir=path_folder))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_folder + 'numer_results.csv', index=False)
# path_dir='out/compare/sensitivity run times/'
# rbt_od_set = post_process.load('out/compare/sensitivity run times/rbt_numer.pkl')
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
# plt.boxplot(rbt_od_set, labels=tags, sym='')
# plt.xticks(rotation=45)
# plt.xlabel('method')
# plt.ylabel('reliability buffer time (min)')
# plt.legend()
# plt.tight_layout()
# plt.savefig(path_dir + 'rbt.png')
# plt.close()


# PART 2 SENSITIVITY TO COMPLIANCE FACTOR
# path_tr_rl0_0 = 'out/DDQN-LA/trajectory_set_0224-124312.pkl'
# path_p_rl0_0 = 'out/DDQN-LA/pax_set_0224-124312.pkl'
# path_tr_rl0_10 = 'out/DDQN-LA/trajectory_set_0225-184023.pkl' # 0225-1827
# path_p_rl0_10 = 'out/DDQN-LA/pax_set_0225-184023.pkl' # 0225-1827
# path_tr_rl0_20 = 'out/DDQN-LA/trajectory_set_0225-181602.pkl' # 0225-1755
# path_p_rl0_20 = 'out/DDQN-LA/pax_set_0225-181602.pkl' # 0225-1755
# path_tr_rl2_0 = 'out/DDQN-HA/trajectory_set_0223-183027.pkl' # 0222-2247
# path_p_rl2_0 = 'out/DDQN-HA/pax_set_0223-183027.pkl'
# path_tr_rl2_10 = 'out/DDQN-HA/trajectory_set_0225-191414.pkl' # 0225-1852
# path_p_rl2_10 = 'out/DDQN-HA/pax_set_0225-191414.pkl' # 0225-1852
# path_tr_rl2_20 = 'out/DDQN-HA/trajectory_set_0225-194326.pkl' # 0225-1934
# path_p_rl2_20 = 'out/DDQN-HA/pax_set_0225-194326.pkl' # 0225-1934
# tags = ['DDQN-LA(0.0)', 'DDQN-HA(0.0)', 'DDQN-LA(0.1)', 'DDQN-HA(0.1)', 'DDQN-LA(0.2)', 'DDQN-HA(0.2)']
# prc = PostProcessor([path_tr_rl0_0, path_tr_rl2_0, path_tr_rl0_10, path_tr_rl2_10, path_tr_rl0_20, path_tr_rl2_20],
#                     [path_p_rl0_0, path_p_rl2_0, path_p_rl0_10, path_p_rl2_10, path_p_rl0_20, path_p_rl2_20], tags)
# path_folder = 'out/compare/sensitivity compliance/'
# results = {}
# results.update(prc.pax_times_fast(path_dir=path_folder))
# results.update(prc.headway(path_dir=path_folder))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_folder + 'numer_results.csv', index=False)
# path_dir='out/compare/sensitivity compliance/'
# rbt_od_set = post_process.load('out/compare/sensitivity compliance/rbt_numer.pkl')
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
# plt.boxplot(rbt_od_set, labels=tags, sym='')
# plt.xticks(rotation=45)
# plt.xlabel('method')
# plt.ylabel('reliability buffer time (min)')
# plt.legend()
# plt.tight_layout()
# plt.savefig(path_dir + 'rbt.png')
# plt.close()
# print("ran in %.2f seconds" % (time.time()-st))
