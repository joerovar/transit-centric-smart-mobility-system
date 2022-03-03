import random
import numpy as np
import pandas as pd
from sim_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl, DetailedSimulationEnvWithDeepRL
from file_paths import *
from input import STOPS
from post_process import save, load, plot_sensitivity_whisker
import matplotlib.pyplot as plt
from datetime import datetime
from output import PostProcessor
st = time.time()


def run_base_detailed(replications=4, save_results=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for _ in range(replications):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)

    if save_results:
        path_trajectories = 'out/NC/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/NC/pax_set_' + tstamp + ext_var
        save(path_trajectories, trajectories_set)
        save(path_completed_pax, pax_set)
    return


def run_base_control_detailed(replications=2, save_results=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for _ in range(replications):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save_results:
        path_trajectories = 'out/EH/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/EH/pax_set_' + tstamp + ext_var
        save(path_trajectories, trajectories_set)
        save(path_completed_pax, pax_set)
    return


def run_sample_rl(episodes=1, simple_reward=False):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    for _ in range(episodes):
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
            env.update_rewards(simple_reward=simple_reward, weight_wait_time=0.3)
            done = env.prep()
    return


# RUN BENCHMARK
# run_base_detailed(replications=70, save_results=True)
# run_base_control_detailed(replications=70, save_results=True)

# BENCHMARK COMPARISON
prc = PostProcessor([path_tr_nc_b, path_tr_eh_b, path_tr_ddqn_la_b, path_tr_ddqn_ha1_b,
                     path_tr_ddqn_ha2_b, path_tr_ddqn_ha3_b, path_tr_ddqn_ha4_b],
                    [path_p_nc_b, path_p_eh_b, path_p_ddqn_la_b, path_p_ddqn_ha1_b,
                     path_p_ddqn_ha2_b, path_p_ddqn_ha3_b, path_p_ddqn_ha4_b], tags_b)
# prc.pax_profile_base()
results = {}
results.update(prc.pax_times_fast(path_dir=path_dir_b, include_rbt=False))

rbt_od_set = load(path_dir_b + 'rbt_numer.pkl')
for i in range(len(rbt_od_set)):
    rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
results.update({'rbt_od': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set]})
results.update(prc.headway(path_dir=path_dir_b))
results_df = pd.DataFrame(results, columns=list(results.keys()))
results_df.to_csv('out/compare/benchmark/numer_results.csv', index=False)

plt.boxplot(rbt_od_set, labels=tags_b, sym='', widths=0.2)
plt.xticks(rotation=45)
plt.xlabel('method')
plt.ylabel('reliability buffer time (min)')
plt.tight_layout()
plt.savefig(path_dir_b + 'rbt.png')
plt.close()

# # VARIABILITY RUN TIMES
#
# prc = PostProcessor([path_tr_ddqn_la_low_s1, path_tr_ddqn_ha_low_s1, path_tr_ddqn_la_base_s1, path_tr_ddqn_ha_base_s1,
#                      path_tr_ddqn_la_high_s1, path_tr_ddqn_ha_high_s1],
#                     [path_p_ddqn_la_low_s1, path_p_ddqn_ha_low_s1, path_p_ddqn_la_base_s1, path_p_ddqn_ha_base_s1,
#                      path_p_ddqn_la_high_s1, path_p_ddqn_ha_high_s1], tags_s1)
# results = {}
# results.update(prc.pax_times_fast(path_dir=path_dir_s1, sensitivity_run_t=True, include_rbt=False))
#
# rbt_od_set = load(path_dir_s1 + 'rbt_numer.pkl')
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
# plot_sensitivity_whisker(rbt_od_set, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
#                          'reliability buffer time (min)', path_dir_s1+'rbt.png')
# results.update({'rbt_od': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set]})
# results.update(prc.headway(path_dir=path_dir_s1, sensitivity_run_t=True))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_dir_s1 + 'numer_results.csv', index=False)
#
# # SENSITIVITY TO COMPLIANCE FACTOR
#
# prc = PostProcessor([path_tr_ddqn_la_base_s2, path_tr_ddqn_ha_base_s2, path_tr_ddqn_la_10_s2, path_tr_ddqn_ha_10_s2,
#                      path_tr_ddqn_la_20_s2, path_tr_ddqn_ha_20_s2],
#                     [path_p_ddqn_la_base_s2, path_p_ddqn_ha_base_s2, path_p_ddqn_la_10_s2, path_p_ddqn_ha_10_s2,
#                      path_p_ddqn_la_20_s2, path_p_ddqn_ha_20_s2], tags_s2)
# results = {}
# results.update(prc.pax_times_fast(path_dir=path_dir_s2, sensitivity_compliance=True, include_rbt=False))
#
# rbt_od_set = load(path_dir_s2 + 'rbt_numer.pkl')
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
# plot_sensitivity_whisker(rbt_od_set, ['DDQN-LA', 'DDQN-HA'], ['0% (base)', '10%', '20%'],
#                          'reliability buffer time (min)', path_dir_s1+'rbt.png')
# results.update({'rbt_od': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set]})
# results.update(prc.headway(path_dir=path_dir_s2, sensitivity_compliance=True))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_dir_s2 + 'numer_results.csv', index=False)


