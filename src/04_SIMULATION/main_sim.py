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


def run_base_control_detailed(replications=2, save_results=False, time_dep_tt=True, time_dep_dem=True,
                              hold_adj_factor=0.0):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for _ in range(replications):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt,
                                               time_dependent_demand=time_dep_dem, hold_adj_factor=hold_adj_factor)
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
            env.update_rewards(simple_reward=simple_reward, weight_ride_time=0.3)
            done = env.prep()
    return


N_REPLICATIONS = 70

# RUN BENCHMARK
# run_base_detailed(replications=N_REPLICATIONS, save_results=True)
# run_base_control_detailed(replications=N_REPLICATIONS, save_results=True, hold_adj_factor=0.8)
# run_base_control_detailed(replications=N_REPLICATIONS, save_results=True, hold_adj_factor=0.6)
# WEIGHTS COMPARISON
# prc_w = PostProcessor([path_tr_ddqn_ha3, path_tr_ddqn_ha5, path_tr_ddqn_ha7, path_tr_ddqn_ha9, path_tr_ddqn_ha11],
#                       [path_p_ddqn_ha3, path_p_ddqn_ha5, path_p_ddqn_ha7, path_p_ddqn_ha9, path_p_ddqn_ha11],
#                       tags_w, N_REPLICATIONS, path_dir_w)
# results_w = {}
# results_w.update(prc_w.pax_times_fast(include_rbt=True))
# rbt_od_set = load(path_dir_w + 'rbt_numer.pkl')
# results_w.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
#                   'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
# prc_w.write_trajectories()
# results_w.update(prc_w.control_actions())
# results_w.update(prc_w.trip_time_dist())
# results_w.update(prc_w.headway())
# results_df = pd.DataFrame(results_w, columns=list(results_w.keys()))
# results_df.to_csv(path_dir_w + 'numer_results.csv', index=False)
# fig, axs = plt.subplots(ncols=2)
# axs[0].boxplot(rbt_od_set, labels=tags_w, sym='', widths=0.2)
# axs[0].set_xticks(np.arange(1, len(tags_w)+1))
# axs[0].set_xticklabels(tags_w, fontsize=8)
# axs[0].tick_params(axis='y', labelsize=8)
# axs[0].set_xlabel(r'$W_{wait}$')
# axs[0].set_ylabel('reliability buffer time (min)', fontsize=9)
#
#
# wt_all_set = load(path_dir_w + 'wt_numer.pkl')
# axs[1].boxplot(wt_all_set, labels=tags_w, sym='', widths=0.2)
# axs[1].set_xticks(np.arange(1, len(tags_w)+1))
# axs[1].set_xticklabels(tags_w, fontsize=8)
# axs[1].tick_params(axis='y', labelsize=8)
# axs[1].set_xlabel(r'$W_{wait}$')
# axs[1].set_ylabel('avg pax wait time (min)', fontsize=9)
#
# plt.tight_layout()
# plt.savefig(path_dir_w + 'pax_times.png')
# plt.close()

# BENCHMARK COMPARISON
# prc = PostProcessor([path_tr_nc_b, path_tr_eh_b, path_tr_ddqn_la_b,
#                      path_tr_ddqn_ha_b],
#                     [path_p_nc_b, path_p_eh_b, path_p_ddqn_la_b,
#                      path_p_ddqn_ha_b], tags_b, N_REPLICATIONS,
#                     path_dir_b)

# prc.sample_trajectories()
# prc.pax_profile_base()
# results = {}
# results.update(prc.pax_times_fast(include_rbt=False))
#
# rbt_od_set = load(path_dir_b + 'rbt_numer.pkl')
# rbt_od_set = rbt_od_set[:3] + [rbt_od_set[-1]]
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt / 60 for rbt in rbt_od_set[i]]
# results.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
#                 'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
# results.update(prc.headway())
# results.update(prc.load_profile())
# # prc.write_trajectories()
# results.update(prc.control_actions())
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_dir_b + 'numer_results.csv', index=False)
#
# #
# fig, axs = plt.subplots(ncols=2)
# axs[0].boxplot(rbt_od_set, labels=tags_b, sym='', widths=0.2)
# axs[0].set_xticks(np.arange(1, len(tags_b) + 1))
# axs[0].set_xticklabels(tags_b, rotation=90, fontsize=8)
# axs[0].tick_params(axis='y', labelsize=8)
# axs[0].set_ylabel('reliability buffer time (min)', fontsize=8)
#
# wt_all_set = load(path_dir_b + 'wt_numer.pkl')
# axs[1].boxplot(wt_all_set, labels=tags_b, sym='', widths=0.2)
# axs[1].set_xticks(np.arange(1, len(tags_b) + 1))
# axs[1].set_xticklabels(tags_b, rotation=90, fontsize=8)
# axs[1].tick_params(axis='y', labelsize=8)
# axs[1].set_ylabel('avg pax wait time (min)', fontsize=8)
#
# plt.tight_layout()
# plt.savefig(path_dir_b + 'pax_times.png')
# plt.close()


# CHECK TRIP TIMES (DWELL TIMES AND EXTREME TT BOUND)
prc_t = PostProcessor([path_tr_nc_t, path_tr_eh_t, path_tr_ddqn_la_t, path_tr_ddqn_ha_t],
                      [path_p_nc_t, path_p_eh_t, path_p_ddqn_la_t, path_p_ddqn_la_t], tags_b, 30, path_dir_b)
prc_t.trip_time_dist()

# # VARIABILITY RUN TIMES
#
# prc = PostProcessor([path_tr_ddqn_la_low_s1, path_tr_ddqn_ha_low_s1, path_tr_ddqn_la_base_s1, path_tr_ddqn_ha_base_s1,
#                      path_tr_ddqn_la_high_s1, path_tr_ddqn_ha_high_s1],
#                     [path_p_ddqn_la_low_s1, path_p_ddqn_ha_low_s1, path_p_ddqn_la_base_s1, path_p_ddqn_ha_base_s1,
#                      path_p_ddqn_la_high_s1, path_p_ddqn_ha_high_s1], tags_s1, N_REPLICATIONS, path_dir_s1)
# results = {}
# results.update(prc.pax_times_fast(include_rbt=False))
#
# rbt_od_set = load(path_dir_s1 + 'rbt_numer.pkl')
# for i in range(len(rbt_od_set)):
#     rbt_od_set[i] = [rbt/60 for rbt in rbt_od_set[i]]
# wt_all_set = load(path_dir_s1 + 'wt_numer.pkl')
# plot_sensitivity_whisker(rbt_od_set, wt_all_set, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
#                          'reliability buffer time (min)', 'avg pax wait time (min)', path_dir_s1 + 'pax_times.png')
# results.update({'rbt_od': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set]})
# results.update(prc.headway(sensitivity_run_t=True))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_dir_s1 + 'numer_results.csv', index=False)
#
# # SENSITIVITY TO COMPLIANCE FACTOR
#
# prc = PostProcessor([path_tr_eh_base_s2, path_tr_ddqn_la_base_s2, path_tr_ddqn_ha_base_s2,
#                      path_tr_eh_80_s2, path_tr_ddqn_la_80_s2, path_tr_ddqn_ha_80_s2,
#                      path_tr_eh_60_s2, path_tr_ddqn_la_60_s2, path_tr_ddqn_ha_60_s2],
#                     [path_p_eh_base_s2, path_p_ddqn_la_base_s2, path_p_ddqn_ha_base_s2,
#                      path_p_eh_80_s2, path_p_ddqn_la_80_s2, path_p_ddqn_ha_80_s2,
#                      path_p_eh_60_s2, path_p_ddqn_la_60_s2, path_p_ddqn_ha_60_s2], tags_s2, N_REPLICATIONS, path_dir_s2)
# results = {}
# results.update(prc.pax_times_fast(include_rbt=False))
# rbt_od_set = load(path_dir_s2 + 'rbt_numer.pkl')
# wt_all_set = load(path_dir_s2 + 'wt_numer.pkl')
# plot_sensitivity_whisker(rbt_od_set, wt_all_set, ['EH', 'DDQN-LA', 'DDQN-HA'], ['base', '0.8', '0.6'],
#                          'reliability buffer time (min)', 'avg pax wait time (min)', path_dir_s2 + 'pax_times.png')
# results.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
#                 'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
# # results.update(prc.headway(sensitivity_compliance=True))
# results_df = pd.DataFrame(results, columns=list(results.keys()))
# results_df.to_csv(path_dir_s2 + 'numer_results.csv', index=False)
