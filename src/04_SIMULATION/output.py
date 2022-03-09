import numpy as np

from input import STOPS, CONTROLLED_STOPS, IDX_ARR_T, IDX_LOAD, IDX_PICK, IDX_DROP, IDX_HOLD_TIME, IDX_DEP_T, \
    TRIP_IDS_IN, SCHED_DEP_IN
from post_process import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags, nr_reps, path_dir):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'brown', 'purple', 'turquoise']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)
            self.nr_reps = nr_reps
            self.path_dir = path_dir

    def pax_times_fast(self, sensitivity_run_t=False, include_rbt=False,
                       sensitivity_compliance=False):
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
                     'wait_t': [np.around(np.mean(wt_set), decimals=2) for wt_set in wt_all_set],
                     'error_wt': [np.around(np.power(1.96, 2) * np.var(wt_set) / np.sqrt(self.nr_reps), decimals=3)
                                  for wt_set in wt_all_set],
                     'denied_per_mil': [round(db * 1000, 2) for db in db_mean],
                     'wait_time_0_2': pc_wt_0_2_set,
                     'wait_time_2_4': pc_wt_2_4_set,
                     'wait_time_4_inf': pc_wt_4_inf_set}
        if include_rbt:
            save(self.path_dir + 'rbt_numer.pkl', rbt_od_set)
        if sensitivity_run_t:
            plot_sensitivity_whisker(wt_all_set, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
                                     'avg pax wait time (min)', self.path_dir + 'wt.png')
        elif sensitivity_compliance:
            plot_sensitivity_whisker(wt_all_set, ['DDQN-LA', 'DDQN-HA'], ['0% (base)', '10%', '20%'],
                                     'avg pax wait time (min)', self.path_dir + 'wt.png')
        else:
            plt.boxplot(wt_all_set, labels=self.cp_tags, sym='', widths=0.2)
            plt.xticks(rotation=45)
            plt.xlabel('method')
            plt.ylabel('avg pax wait time (min)')
            plt.tight_layout()
            plt.savefig(self.path_dir + 'wt.png')
            plt.close()
        return results_d

    def headway(self, sensitivity_run_t=False, sensitivity_compliance=False):
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
        plot_headway(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname=self.path_dir + 'hw.png',
                     controlled_stops=CONTROLLED_STOPS[:-1])
        results_hw = {'mean_cv_hw_tp': [np.around(np.mean(cv), decimals=2) for cv in cv_hw_tp_set],
                      'error_cv_hw_tp': [np.around(np.power(1.96, 2) * np.var(cv) / np.sqrt(self.nr_reps), decimals=3)
                                         for cv in cv_hw_tp_set],
                      'hw_peak': [np.around(np.mean(hw), decimals=2) for hw in hw_peak_set],
                      'std_hw_peak': [np.around(np.std(hw), decimals=2) for hw in hw_peak_set]}

        if sensitivity_run_t:
            plot_sensitivity_whisker(cv_hw_tp_set, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
                                     'coefficient of variation of headway', self.path_dir + 'hw_bplot.png')
        elif sensitivity_compliance:
            plot_sensitivity_whisker(cv_hw_tp_set, ['DDQN-LA', 'DDQN-HA'], ['0% (base)', '10%', '20%'],
                                     'coefficient of variation of headway', self.path_dir + 'hw_bplot.png')
        else:
            plt.boxplot(cv_hw_tp_set, labels=self.cp_tags, sym='', widths=0.2)
            plt.xticks(rotation=45)
            plt.xlabel('method')
            plt.ylabel('coefficient of variation of headway')
            plt.tight_layout()
            plt.savefig(self.path_dir + 'cv_hw.png')
            plt.close()

            # hw_peak_set = hw_peak_set[:3] + [hw_peak_set[-1]]
            # tags = self.cp_tags[:3] + [self.cp_tags[-1]]
            # plt.boxplot(hw_peak_set, labels=tags, sym='', widths=0.2)
            # plt.xticks(rotation=45)
            # plt.xlabel('method')
            # plt.ylabel('headway (mins)')
            # plt.tight_layout()
            # plt.savefig(self.path_dir + 'peak_hw.png')
            # plt.close()
        return results_hw

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        peak_load_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, temp_peak_loads = load_from_trajectory_set(trips, STOPS, IDX_LOAD, STOPS[57])
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
            peak_load_set.append(temp_peak_loads)
        # plot_load_profile_benchmark(load_profile_set, STOPS, self.cp_tags, self.colors,
        #                             pathname=self.path_dir + 'lp.png', controlled_stops=CONTROLLED_STOPS,
        #                             x_y_lbls=['stop id', 'avg load per trip'], load_sd_set=lp_std_set)
        # peak_load_set = peak_load_set[:3] + [peak_load_set[-1]]
        # tags = self.cp_tags[:3] + [self.cp_tags[-1]]
        # plt.boxplot(peak_load_set, labels=tags, sym='', widths=0.2)
        # plt.xticks(rotation=45)
        # plt.xlabel('method')
        # plt.ylabel('peak load')
        # plt.tight_layout()
        # plt.savefig(self.path_dir + 'peak_load.png')
        # plt.close()

        results_load = {'load_mean': [np.around(np.mean(peak_load), decimals=2) for peak_load in peak_load_set],
                        'std_load': [np.around(np.std(peak_load), decimals=2) for peak_load in peak_load_set]}
        return results_load

    def sample_trajectories(self):
        for i in range(len(self.cp_trips)):
            trips = self.cp_trips[i][35:38]
            # trip_df = pd.read_csv('out/trajectories' + self.cp_tags[i] + '.csv')
            # trip_df = trip_df[trip_df['replication'] == 1]
            plot_trajectories(trips, IDX_ARR_T, IDX_DEP_T, 'out/trajectories' + self.cp_tags[i] + '.png',
                              STOPS, controlled_stops=CONTROLLED_STOPS)
        return

    def write_trajectories(self, only_nc=False):
        i = 0
        compare_trips = self.cp_trips
        if only_nc:
            compare_trips = [self.cp_trips[0]]
        for trips in compare_trips:
            write_trajectory_set(trips, 'out/trajectories' + self.cp_tags[i] +'.csv', IDX_ARR_T, IDX_DEP_T,
                                 IDX_HOLD_TIME,
                                 header=['trip_id', 'stop_id', 'arr_t', 'dep_t', 'pax_load', 'ons', 'offs', 'denied',
                                         'hold_time', 'skipped', 'replication', 'arr_sec', 'dep_sec', 'dwell_sec'])
            i += 1
        return

    def trip_time_dist(self):
        trip_time_mean_set = []
        trip_time_sd_set = []
        for trips in self.cp_trips:
            temp_trip_t = trip_time_from_trajectory_set(trips, IDX_DEP_T, IDX_ARR_T)
            trip_time_mean_set.append(np.around(np.mean(temp_trip_t), decimals=2))
            trip_time_sd_set.append(np.around(np.std(temp_trip_t), decimals=2))
        results_tt = {'tt_mean': trip_time_mean_set,
                      'tt_sd': trip_time_sd_set}
        return results_tt

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
        through = through.tolist()
        plot_pax_profile(ons, offs, lp, STOPS, through, pathname='in/vis/pax_profile_base.png',
                         x_y_lbls=['stop', 'passengers (per trip)',
                                   'through passengers and passenger load (per trip)'],
                         controlled_stops=CONTROLLED_STOPS[:-1])
        return
