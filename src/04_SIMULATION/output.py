import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from input import STOPS, CONTROLLED_STOPS, IDX_ARR_T, IDX_LOAD, IDX_PICK, IDX_DROP, IDX_HOLD_TIME, IDX_DEP_T, \
    TRIP_IDS_IN, SCHED_DEP_IN, IDX_RT_PROGRESS, IDX_FW_H, IDX_BW_H, N_ACTIONS_RL, CONTROL_MEAN_HW, IDX_DENIED, IDX_LOAD_RL, IDX_SKIPPED, FOCUS_TRIPS
from post_process import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags, nr_reps, path_dir):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'brown', 'purple', 'turquoise',
                       'gray']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)
            self.nr_reps = nr_reps
            self.path_dir = path_dir

    def pax_times_fast(self, include_rbt=False):
        db_mean = []
        dbwt_mean = []
        wt_all_set = []
        rbt_od_set = []
        rbt_od_mean = []
        pc_wt_0_2_set = []
        pc_wt_2_4_set = []
        pc_wt_4_inf_set = []
        for pax in self.cp_pax:
            wt_set, dbm, dbwt, rbt_od, pc_wt_0_2, pc_wt_2_4, pc_wt_4_inf = get_pax_times_fast(pax,
                                                                                              len(STOPS),
                                                                                              include_rbt=include_rbt)
            wt_all_set.append(wt_set)
            db_mean.append(dbm)
            dbwt_mean.append(dbwt)
            rbt_od_set.append(rbt_od)
            rbt_od_mean.append(np.mean(rbt_od))
            pc_wt_0_2_set.append(pc_wt_0_2)
            pc_wt_2_4_set.append(pc_wt_2_4)
            pc_wt_4_inf_set.append(pc_wt_4_inf)
        results_d = {'method': self.cp_tags,
                     'wt_mean': [np.around(np.mean(wt_s), decimals=2) for wt_s in wt_all_set],
                     'err_wt_mean': [np.around(np.power(1.96, 2) * np.var(wt_s) / np.sqrt(self.nr_reps), decimals=3)
                                     for wt_s in wt_all_set],
                     'wt_median': [np.around(np.median(wt_s), decimals=2) for wt_s in wt_all_set],
                     'denied_per_mil': [round(db * 1000, 2) for db in db_mean],
                     'wt_0_2': pc_wt_0_2_set,
                     'wt_2_4': pc_wt_2_4_set,
                     'wt_4_inf': pc_wt_4_inf_set}
        if include_rbt:
            save(self.path_dir + 'rbt_numer.pkl', rbt_od_set)
        save(self.path_dir + 'wt_numer.pkl', wt_all_set)

        return results_d

    def headway(self, plot_bars=False):
        cv_hw_set = []
        cv_all_reps = []
        cv_hw_tp_set = []
        hw_peak_set = []
        for trips in self.cp_trips:
            temp_cv_hw, cv_hw_tp, cv_hw_mean, hw_peak = get_headway_from_trajectory_set(trips, IDX_ARR_T, STOPS,
                                                                                        STOPS[50],
                                                                                        controlled_stops=CONTROLLED_STOPS)
            cv_hw_tp_set.append(cv_hw_tp)
            cv_hw_set.append(temp_cv_hw)
            cv_all_reps.append(cv_hw_mean)
            hw_peak_set.append(hw_peak)
        if len(self.cp_tags) <= len(self.colors):
            plot_headway(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname=self.path_dir + 'hw.png',
                         controlled_stops=CONTROLLED_STOPS[:-1])

        results_hw = {'cv_h_tp': [np.around(np.mean(cv), decimals=2) for cv in cv_hw_tp_set],
                      'err_cv_h_tp': [np.around(np.power(1.96, 2) * np.var(cv) / np.sqrt(self.nr_reps), decimals=3)
                                      for cv in cv_hw_tp_set],
                      'h_pk': [np.around(np.mean(hw), decimals=2) for hw in hw_peak_set],
                      'std_h_pk': [np.around(np.std(hw), decimals=2) for hw in hw_peak_set],
                      '95_h_pk': [np.around(np.percentile(hw, 95), decimals=2) for hw in hw_peak_set]}
        # cv_hw_tp_set is for whisker plot
        if plot_bars:
            # cv_hw_set_sub = cv_hw_set[0:3] + [cv_hw_set[-1]]
            # tags = self.cp_tags[0:3] + [self.cp_tags[-1]]
            tags = self.cp_tags
            idx_control_stops = [STOPS.index(cs) + 1 for cs in CONTROLLED_STOPS[:-1]]
            cv_tp_set = []
            for cv in cv_hw_set:
                cv_tp_set.append([cv[k] for k in idx_control_stops])
            x = np.arange(len(idx_control_stops))
            width = 0.1
            fig, ax = plt.subplots()
            print(cv_tp_set)
            bar1 = ax.bar(x - 3 * width / 2, cv_tp_set[0], width, label=tags[0], color='white', edgecolor='black')
            bar2 = ax.bar(x - width / 2, cv_tp_set[1], width, label=tags[1], color='silver', edgecolor='black')
            bar3 = ax.bar(x + width / 2, cv_tp_set[2], width, label=tags[2], color='gray', edgecolor='black')
            bar4 = ax.bar(x + 3 * width / 2, cv_tp_set[3], width, label=tags[3], color='black', edgecolor='black')

            ax.set_ylabel('coefficient of variation of headway')
            ax.set_xlabel('control stop')
            ax.set_xticks(x, idx_control_stops)
            ax.legend()

            fig.tight_layout()
            plt.savefig(self.path_dir + 'cv_hw_bar.png')
            plt.close()

        return results_hw

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        peak_load_set = []
        max_load_set = []
        min_load_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, temp_peak_loads, max_load, min_load = load_from_trajectory_set(trips,
                                                                                                 STOPS, IDX_LOAD,
                                                                                                 STOPS[56])
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
            peak_load_set.append(temp_peak_loads)
            max_load_set.append(max_load)
            min_load_set.append(min_load)
        plot_load_profile_grid(load_profile_set, max_load_set, min_load_set, STOPS, self.cp_tags,
                               pathname=self.path_dir + 'lp_grid.png')
        # plot_load_profile_benchmark(load_profile_set, STOPS, self.cp_tags, self.colors,
        #                             pathname=self.path_dir + 'lp.png', controlled_stops=CONTROLLED_STOPS,
        #                             x_y_lbls=['stop id', 'avg load per trip'], load_sd_set=lp_std_set)

        results_load = {'load_mean': [np.around(np.mean(peak_load), decimals=2) for peak_load in peak_load_set],
                        'std_load': [np.around(np.std(peak_load), decimals=2) for peak_load in peak_load_set],
                        '95_load': [np.around(np.percentile(peak_load, 95), decimals=2) for peak_load in peak_load_set]}
        return results_load

    def sample_trajectories(self):
        for i in range(len(self.cp_trips)):
            trips = self.cp_trips[i][35:38]
            plot_trajectories(trips, IDX_ARR_T, IDX_DEP_T, 'out/trajectories/' + self.cp_tags[i] + '.png',
                              STOPS, controlled_stops=CONTROLLED_STOPS)
        return

    def write_trajectories(self, only_nc=False):
        i = 0
        compare_trips = self.cp_trips
        if only_nc:
            compare_trips = [self.cp_trips[0]]
        for trips in compare_trips:
            write_trajectory_set(trips, 'out/trajectories/' + self.cp_tags[i] + '.csv', IDX_ARR_T, IDX_DEP_T,
                                 IDX_HOLD_TIME,
                                 header=['trip_id', 'stop_id', 'arr_t', 'dep_t', 'pax_load', 'ons', 'offs', 'denied',
                                         'hold_time', 'skipped', 'replication', 'arr_sec', 'dep_sec', 'dwell_sec'])
            i += 1
        return

    def control_actions(self):
        ht_ordered_freq_set = []
        skip_freq_set = []
        bounds = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, np.inf)]
        for i in range(len(self.cp_tags)):
            ht_dist, skip_freq = control_from_trajectory_set('out/trajectories/' + self.cp_tags[i] + '.csv',
                                                             CONTROLLED_STOPS)
            skip_freq_set.append(round(skip_freq, 1))
            ht_dist_arr = np.array(ht_dist)
            ht_ordered_freq = []
            for b in bounds:
                ht_ordered_freq.append(ht_dist_arr[(ht_dist_arr >= b[0]) & (ht_dist_arr < b[1])].size)
            ht_ordered_freq_set.append(ht_ordered_freq)
        results = {'skip_freq': skip_freq_set}
        return results

    def trip_times(self, keep_nc=False, plot=False):
        all_trip_times = []
        trip_time_mean_set = []
        trip_time_sd_set = []
        trip_time_95_set = []
        trip_time_85_set = []
        i = 0
        for trips in self.cp_trips:
            temp_trip_t = trip_time_from_trajectory_set(trips, IDX_DEP_T, IDX_ARR_T)
            all_trip_times.append(temp_trip_t)
            if i == 0 and keep_nc:
                temp_trip_t = trip_time_from_trajectory_set(self.cp_trips[0], IDX_DEP_T, IDX_ARR_T)
                save(self.path_dir + 'trip_t_sim.pkl', temp_trip_t)
            trip_time_mean_set.append(np.around(np.mean(temp_trip_t) / 60, decimals=2))
            trip_time_sd_set.append(np.around(np.std(temp_trip_t) / 60, decimals=2))
            trip_time_95_set.append(np.around(np.percentile(temp_trip_t, 95) / 60, decimals=2))
            trip_time_85_set.append(np.around(np.percentile(temp_trip_t, 90) / 60, decimals=2))
            i += 1
        if plot and len(all_trip_times) == 4:
            plot_4_trip_t_dist(all_trip_times, self.cp_tags, self.path_dir)
        save(self.path_dir + 'all_trip_t.pkl', all_trip_times)
        results_tt = {'tt_mean': trip_time_mean_set,
                      'tt_sd': trip_time_sd_set,
                      'tt_95': trip_time_95_set,
                      'tt_85': trip_time_85_set}
        return results_tt

    def validation(self):
        temp_cv_hw, cv_hw_tp, cv_hw_mean, hw_peak = get_headway_from_trajectory_set(self.cp_trips[0], IDX_ARR_T, STOPS,
                                                                                    STOPS[50],
                                                                                    controlled_stops=CONTROLLED_STOPS)
        cv_hw_set = [cv_hw_tp]
        plot_headway(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname=self.path_dir + 'hw.png',
                     controlled_stops=CONTROLLED_STOPS[:-1])
        return

    def load_profile_validation(self):
        load_profile_set = []
        lp_std_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, _, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
        lp_input = load('in/xtr/rt_20-2019-09/load_profile.pkl')
        load_profile_set.append(lp_input)
        plot_load_profile_benchmark(load_profile_set, STOPS, ['simulated', 'observed'], self.colors,
                                    pathname='out/validation/lp.png', x_y_lbls=['stop id', 'avg load per trip'])
        return

    def departure_delay_validation(self):
        dep_delay_in_input = load('in/xtr/rt_20-2019-09/dep_delay_in.pkl')
        dep_delay_in_simul = get_departure_delay(self.cp_trips[0], IDX_DEP_T, TRIP_IDS_IN, SCHED_DEP_IN)
        sns.kdeplot(dep_delay_in_input, label='observed')
        sns.kdeplot(dep_delay_in_simul, label='simulated')
        plt.legend()
        plt.savefig('out/validation/dep_delay.png')
        plt.close()
        return

    def pax_profile_base(self):
        lp, _, _, ons, offs = pax_per_trip_from_trajectory_set(self.cp_trips[0], IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
        through = np.subtract(lp, offs)
        through[through < 0] = 0
        through = through.tolist()
        plot_pax_profile(ons, offs, lp, STOPS, through, pathname='in/vis/pax_profile_base.png',
                         x_y_lbls=['stop', 'passengers (per trip)',
                                   'through passengers and passenger load (per trip)'],
                         controlled_stops=CONTROLLED_STOPS[:-1])
        return


def count_load(file_dir, hw_threshold, count_skip=False):
    cs_load = []
    peak_load = []
    cs_hw = []
    peak_hw = []
    trajectory_set = load(file_dir)
    activate = False
    skipped = 0
    not_skipped = 0
    for trajectory in trajectory_set:
        last_t = None
        last_t_pk = None
        for trip in trajectory:
            for stop_info in trajectory[trip]:
                if stop_info[0] == STOPS[47]:
                    arr_t = stop_info[IDX_ARR_T]
                    if last_t:
                        hw = arr_t - last_t
                        if hw > hw_threshold:
                            if count_skip:
                                if stop_info[IDX_SKIPPED]:
                                    skipped += 1
                                else:
                                    not_skipped += 1
                            cs_load.append(stop_info[IDX_LOAD])
                            cs_hw.append(hw)
                            activate = True
                        else:
                            activate = False
                    last_t = deepcopy(arr_t)
                if activate and stop_info[0] == STOPS[56]:
                    peak_load.append(stop_info[IDX_LOAD])
                    arr_t = stop_info[IDX_ARR_T]
                    peak_hw.append(arr_t-last_t_pk)
                    activate = False
                if stop_info[0] == STOPS[56]:
                    last_t_pk = stop_info[IDX_ARR_T]

    avg_pk_load = np.around(np.mean(peak_load), decimals=2)
    avg_pk_hw = np.around(np.mean(peak_hw)/60, decimals=2)
    avg_prev_hw = np.around(np.mean(cs_hw)/60, decimals=2)
    avg_prev_load = np.around(np.mean(cs_load), decimals=2)
    nr_trips = len(peak_load)
    nr_trips_prev = len(cs_load)
    # print(nr_trips, nr_trips_prev)
    if count_skip:
        skipped_freq = round(skipped / (skipped + not_skipped)*100, 2)
        return avg_prev_hw, avg_prev_load, avg_pk_hw, avg_pk_load, skipped_freq
    else:
        return avg_prev_hw, avg_prev_load, avg_pk_hw, avg_pk_load


def policy():
    sars_set = load('out/DDQN-HA/0329-165457-sars_set.pkl')
    n_stops = len(STOPS)
    rt_progress = [STOPS.index(c) / n_stops for c in CONTROLLED_STOPS[:-1]]
    control_stops = [STOPS.index(c) + 1 for c in CONTROLLED_STOPS[:-1]]
    d = {'stop': [], 'fw_h': [], 'bw_h': [], 'action': []}
    d2 = {'stop': [], 'fw_h': [], 'load': [], 'action': []}
    for sars_rep in sars_set:
        for trip in sars_rep:
            for sars in sars_rep[trip]:
                d['stop'].append(round(sars[0][IDX_RT_PROGRESS] * n_stops) + 1)
                d['fw_h'].append(sars[0][IDX_FW_H])
                d['bw_h'].append(sars[0][IDX_BW_H])
                d['action'].append(sars[1])
                if sars[1] == 0 and trip in FOCUS_TRIPS:
                    d2['stop'].append(round(sars[0][0]*n_stops+1))
                    d2['fw_h'].append(sars[0][IDX_FW_H])
                    d2['load'].append(sars[0][IDX_LOAD_RL])
                    d2['action'].append(sars[1])

    denied_df = pd.DataFrame(d2)

    d_subset = denied_df[denied_df['stop'] == 48]
    freq_skip_end = round(d_subset.shape[0]/denied_df.shape[0] * 100, 2)
    freq_skip_start = round((denied_df.shape[0] - d_subset.shape[0]) / denied_df.shape[0] * 100, 2)
    avg_load = round(d_subset['load'].mean(), 2)
    avg_fw_h = round(d_subset['fw_h'].mean()/60, 2)

    denied_df.to_csv('out/compare/denied_policy.csv', index=False)
    trajectory_set = load('out/DDQN-HA/0329-165457-trajectory_set.pkl')
    skipped_trips = []
    reps = []
    peak_load = []

    hw_threshold = 7*60
    eh_prev_hw, eh_prev_load, eh_pk_hw, eh_pk_load = count_load('out/EH/0329-155402-trajectory_set.pkl', hw_threshold)
    ddqn_la_prev_hw, ddqn_la_prev_load, ddqn_la_pk_hw, ddqn_la_pk_load = count_load('out/DDQN-LA/0329-185304-trajectory_set.pkl', hw_threshold)
    ddqn_ha_prev_hw, ddqn_ha_prev_load, ddqn_ha_pk_hw, ddqn_ha_pk_load, skipped_freq = count_load('out/DDQN-HA/0405-014450-trajectory_set.pkl', hw_threshold, count_skip=True)

    print(eh_prev_hw, eh_prev_load, eh_pk_hw, eh_pk_load)
    print(ddqn_la_prev_hw, ddqn_la_prev_load, ddqn_la_pk_hw, ddqn_la_pk_load)
    print(ddqn_ha_prev_hw, ddqn_ha_prev_load, ddqn_ha_pk_hw, ddqn_ha_pk_load, skipped_freq)
    # for trajectory in trajectory_set:
    #     for trip in trajectory:
    #         for stop_info in trajectory[trip]:
    #             if activate and stop_info[0] == STOPS[56]:
    #                 peak_load.append(stop_info[IDX_LOAD])
    #                 activate = False
    #             elif stop_info[IDX_SKIPPED] == 1 and stop_info[0] == CONTROLLED_STOPS[-2]:
    #                 activate = True
    # avg_peak_load = np.around(np.mean(peak_load), decimals=2)
    # print(freq_skip_end, freq_skip_start, avg_load, avg_peak_load, avg_fw_h)


    # policy_df = pd.DataFrame(d)
    # policy_df['delta_h'] = (policy_df['fw_h'] - policy_df['bw_h']) / CONTROL_MEAN_HW
    # policy_on_t_df = policy_df[(policy_df['delta_h'] < 0.4) & (policy_df['delta_h'] > 0.2)]
    # policy_early_df = policy_df[policy_df['delta_h'] < -0.7]
    # policy_late_df = policy_df[policy_df['delta_h'] > 0.7]
    # actions = [i for i in range(N_ACTIONS_RL)]
    # on_t = []
    # early = []
    # late = []
    # for (policy, lst) in ((policy_on_t_df, on_t), (policy_early_df, early), (policy_late_df, late)):
    #     for action in actions:
    #         action_df = policy[policy['action'] == action]
    #         action_lst = []
    #         for stop in control_stops:
    #             count_action = action_df[action_df['stop'] == stop]['action'].shape[0]
    #             action_lst.append(count_action)
    #         lst.append(action_lst)
    # on_t_arr = np.array(on_t)
    # early_arr = np.array(early)
    # late_arr = np.array(late)
    # for colum in range(on_t_arr.shape[1]):
    #     on_t_arr[:, colum] = on_t_arr[:, colum] / on_t_arr[:, colum].sum() * 100
    #     early_arr[:, colum] = early_arr[:, colum] / early_arr[:, colum].sum() * 100
    #     late_arr[:, colum] = late_arr[:, colum] / late_arr[:, colum].sum() * 100
    # fig, ax = plt.subplots(ncols=3, sharey='all', sharex='all')
    # mesh1 = ax[0].pcolormesh(early_arr, cmap='Greys')
    # mesh1.set_clim(0, 100)
    # mesh2 = ax[1].pcolormesh(on_t_arr, cmap='Greys')
    # mesh2.set_clim(0, 100)
    # mesh3 = ax[2].pcolormesh(late_arr, cmap='Greys')
    # mesh3.set_clim(0, 100)
    # plt.show()
    # plt.close()
    return


# policy()
