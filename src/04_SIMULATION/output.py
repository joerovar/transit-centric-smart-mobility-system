from input import *
from post_process import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'brown', 'purple', 'turquoise']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)

    def pax_times_fast(self, path_dir='out/compare/benchmark/', sensitivity_run_t=False, include_rbt=False,
                       sensitivity_compliance=False):
        db_mean = []
        dbwt_mean = []
        wt_all_set = []
        rbt_od_set = []
        rbt_od_mean = []
        for pax in self.cp_pax:
            wt_set, dbm, dbwt, rbt_od = get_pax_times_fast(pax, len(STOPS), include_rbt=include_rbt)
            wt_all_set.append(wt_set)
            db_mean.append(dbm)
            dbwt_mean.append(dbwt)
            rbt_od_set.append(rbt_od)
            rbt_od_mean.append(np.mean(rbt_od))
        results_d = {'method': self.cp_tags,
                     'wait_t': [np.around(np.mean(wt_set), decimals=2) for wt_set in wt_all_set],
                     'error_wt': [np.around(np.std(wt_set), decimals=2) for wt_set in wt_all_set],
                     'denied_per_mil': [round(db * 1000, 2) for db in db_mean]}
        if include_rbt:
            save(path_dir + 'rbt_numer.pkl', rbt_od_set)
        if sensitivity_run_t:
            plot_sensitivity_whisker(wt_all_set, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
                                     'avg pax wait time (min)', path_dir + 'wt.png')
        elif sensitivity_compliance:
            plot_sensitivity_whisker(wt_all_set, ['DDQN-LA', 'DDQN-HA'], ['0% (base)', '10%', '20%'],
                                     'avg pax wait time (min)', path_dir + 'wt.png')
        else:
            plt.boxplot(wt_all_set, labels=self.cp_tags, sym='')
            plt.xticks(rotation=45)
            plt.xlabel('method')
            plt.ylabel('avg pax wait time (min)')
            plt.tight_layout()
            plt.savefig(path_dir + 'wt.png')
            plt.close()
        return results_d

    def headway(self, path_dir='out/compare/benchmark/', sensitivity_run_t=False, sensitivity_compliance=False):
        cv_hw_set = []
        cv_all_reps = []
        for trips in self.cp_trips:
            temp_cv_hw, hw_at_tp, cv_hw_mean = get_headway_from_trajectory_set(trips, IDX_ARR_T, STOPS,
                                                                               controlled_stops=CONTROLLED_STOPS)
            cv_hw_set.append(temp_cv_hw)
            cv_all_reps.append(cv_hw_mean)
        plot_headway(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname=path_dir + 'hw.png',
                     controlled_stops=CONTROLLED_STOPS[:-1])

        results_hw = {'mean_cv_hw': [np.around(np.mean(cv), decimals=2) for cv in cv_hw_set],
                      'error_cv_hw': [np.around(np.std(cv), decimals=2) for cv in cv_hw_set]}

        if sensitivity_run_t:
            plot_sensitivity_whisker(cv_all_reps, ['DDQN-LA', 'DDQN-HA'], ['cv: -20%', 'cv: base', 'cv: +20%'],
                                     'coefficient of variation of headway', path_dir + 'hw_bplot.png')
        elif sensitivity_compliance:
            plot_sensitivity_whisker(cv_all_reps, ['DDQN-LA', 'DDQN-HA'], ['0% (base)', '10%', '20%'],
                                     'coefficient of variation of headway', path_dir + 'hw_bplot.png')
        else:
            plt.boxplot(cv_all_reps, labels=self.cp_tags, sym='')
            plt.xticks(rotation=45)
            plt.xlabel('method')
            plt.ylabel('coefficient of variation of headway')
            plt.tight_layout()
            plt.savefig(path_dir + 'hw_bplot.png')
            plt.close()
        return results_hw

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, _, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
        plot_load_profile_benchmark(load_profile_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/lp.png',
                                    controlled_stops=CONTROLLED_STOPS, x_y_lbls=['stop id', 'avg load per trip'])
        avg_lp_sd = [np.mean(lp_sd) for lp_sd in lp_std_set]
        print(f'avg load sd {avg_lp_sd}')
        idx_control_stops = [STOPS.index(cs) for cs in CONTROLLED_STOPS]
        lp_cs = []
        for lp in load_profile_set:
            lp_cs.append([])
            for i in idx_control_stops:
                lp_cs[-1].append(lp[i])
        print(f'load per control stop {lp_cs}')
        return

    def write_trajectories(self, only_nc=False):
        i = 0
        compare_trips = self.cp_trips
        if only_nc:
            compare_trips = [self.cp_trips[0]]
        for trips in compare_trips:
            write_trajectory_set(trips, 'out/' + self.cp_tags[i] + '/trajectories.csv', IDX_ARR_T, IDX_DEP_T,
                                 IDX_HOLD_TIME,
                                 header=['trip_id', 'stop_id', 'arr_t', 'dep_t', 'pax_load', 'ons', 'offs', 'denied',
                                         'hold_time', 'skipped', 'replication', 'arr_sec', 'dep_sec', 'dwell_sec'])
            i += 1
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
        through = through.tolist()
        plot_pax_profile(ons, offs, lp, STOPS, through, pathname='in/vis/pax_profile_base.png',
                         x_y_lbls=['stop', 'passengers (per trip)',
                                   'through passengers and passenger load (per trip)'],
                         controlled_stops=CONTROLLED_STOPS[:-1])
        return
