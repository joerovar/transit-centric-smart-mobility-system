import matplotlib.pyplot as plt
from input import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)

    def write_trajectories(self):
        i = 0
        for trips in self.cp_trips:
            write_trajectory_set(trips, 'out/trajectories' + str(i) + '.csv', IDX_ARR_T, IDX_DEP_T, IDX_HOLD_TIME,
                                 header=['trip_id', 'stop_id', 'arr_t', 'dep_t', 'pax_load', 'ons', 'offs', 'denied',
                                         'hold_time', 'skipped', 'replication'])
            i += 1
        return

    def headway(self):
        cv_hw_set = []
        sd_hw_at_tp = []
        for trips in self.cp_trips:
            temp_cv_hw, hw_at_tp = get_headway_from_trajectory_set(trips, IDX_ARR_T, IDX_DEP_T, STOPS,
                                                                   controlled_stops=CONTROLLED_STOPS)
            cv_hw_set.append(temp_cv_hw)
            sd_hw_at_tp.append(np.array(hw_at_tp).std())
        # plot_headway_benchmark(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/hw.png',
        #                        controlled_stops=CONTROLLED_STOPS)
        print(print(f'mean cv headway {[np.mean(cv) for cv in cv_hw_set]}'))
        return

    def total_trip_time_distribution(self):
        trip_time_set = []
        for trip in self.cp_trips:
            trip_time = tot_trip_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            trip_time_set.append(trip_time)
        std_run_times = []
        extr_run_times = []
        mean_run_times = []
        for t in trip_time_set:
            t_arr = np.array(t)
            mean_run_times.append(t_arr.mean())
            std_run_times.append(t_arr.std())
            extr_run_times.append(np.percentile(t_arr, 95))
        # print(mean_run_times)
        # print(std_run_times)
        # print(extr_run_times)
        # plot_travel_time_benchmark(trip_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ttd.png')
        print(f'trip time percentile {[np.percentile(t, 90) for t in trip_time_set]}')
        return

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, _, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
        # print(lp_std_set[0])
        # print(load_profile_set[0])
        plot_load_profile_benchmark(load_profile_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/lp.png',
                                    controlled_stops=CONTROLLED_STOPS, x_y_lbls=['stop id', 'avg load per trip'])
        plot_load_profile_benchmark(lp_std_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/lp_sd.png',
                                    controlled_stops=CONTROLLED_STOPS, x_y_lbls=['stop id', 'sd load per trip'])
        return

    def hold_time(self):
        hold_time_set = []
        hold_time_dist_set = []
        for trips in self.cp_trips:
            ht_per_stop, ht_per_trip, ht_dist_per_trip = hold_time_from_trajectory_set(trips, IDX_HOLD_TIME)
            hold_time_set.append(ht_per_trip)
            hold_time_dist_set.append(ht_dist_per_trip)
        # plot_mean_hold_time_benchmark(hold_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ht.png')
        # plot_hold_time_distribution_benchmark(hold_time_dist_set, self.cp_tags, self.colors,
        #                                       pathname='out/benchmark/ht_dist.png')
        print(f'hold time mean {[np.nanmean(ht) for ht in hold_time_dist_set]}')
        return

    def denied(self):
        denied_set = []
        for trips in self.cp_trips:
            _, _, tot_ons, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            denied_set.append(denied_from_trajectory_set(trips, IDX_DENIED, tot_ons))
        plot_denied_benchmark(denied_set, self.cp_tags, self.colors, pathname='out/benchmark/db.png')
        return

    def wait_times(self):
        wait_time_set = []
        counter = 0
        for pax in self.cp_pax:
            wt_mean = get_wait_times(pax, STOPS)
            wait_time_set.append(wt_mean)
            counter += 1
        scheduled_avg_wait = FOCUS_TRIPS_MEAN_HW / 2
        # plot_wait_time_benchmark(wait_time_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/wt.png',
        #                          scheduled_wait=scheduled_avg_wait, controlled_stops=CONTROLLED_STOPS)
        print(f'wait time {[np.nanmean(wt) for wt in wait_time_set]}')
        print(f'wait time percentile {[np.nanpercentile(wt, 90) for wt in wait_time_set]}')
        return

    def rbt_difference(self, a, b):
        idx1 = self.cp_tags.index(a)
        idx2 = self.cp_tags.index(b)
        _, _, rbt1, _, _, _ = get_journey_times(self.cp_pax[idx1], STOPS)
        _, _, rbt2, _, _, _ = get_journey_times(self.cp_pax[idx2], STOPS)
        plot_difference_od(rbt1 - rbt2, STOPS, controlled_stops=CONTROLLED_STOPS,
                           pathname='out/benchmark/rbt_diff_' + self.cp_tags[idx1] + '_' + self.cp_tags[idx2] + '.png',
                           clim=(-250, 250))
        return

    def journey_times(self):
        jt_sums = []
        extr_jt_sums = []
        for pax in self.cp_pax:
            _, _, _, _, jt_sum, extr_jt_sum = get_journey_times(pax, STOPS)
            jt_sums.append(jt_sum)
            extr_jt_sums.append(extr_jt_sum)
        return

    # def params_for_policy(self, tag='RL'):
    #     idx = self.cp_tags.index(tag)
    #     mean_load_comb, _, ons_comb, _, _, _ = pax_per_trip_from_trajectory_set(self.cp_trips[idx], IDX_LOAD,
    #                                                                             IDX_PICK, IDX_DROP, STOPS)
    #     return mean_load_comb, ons_comb

    def dwell_time_validation(self):
        dt_set = []
        dt_std_set = []
        dt_tot_set = []
        for trip in self.cp_trips:
            _, _, dt_mean, dt_std, dt_tot = travel_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            dt_set.append(dt_mean)
            dt_std_set.append(dt_std)
            dt_tot_set.append(dt_tot)
        paths = ('out/benchmark/dwell_t_mean.png', 'out/benchmark/dwell_t_std.png', 'out/benchmark/dwell_t_tot.png')
        dt_input_mean = load('in/xtr/rt_20-2019-09/dwell_times_mean.pkl')
        dt_input_std = load('in/xtr/rt_20-2019-09/dwell_times_std.pkl')
        dt_input_tot = load('in/xtr/rt_20-2019-09/dwell_times_tot.pkl')
        dt_set.append(dt_input_mean)
        dt_std_set.append(dt_input_std)
        dt_tot_set.append(dt_input_tot)
        tags = ['simulated', 'observed']
        i = 0
        for d in dt_set:
            plt.plot([i for i in range(1, len(d) + 1)], d.values(), color=self.colors[i], label=tags[i])
            i += 1
        plt.ylabel('mean dwell time (seconds)')
        plt.xlabel('stop number')
        plt.legend()
        plt.savefig(paths[0])
        plt.close()

        i = 0
        for d in dt_std_set:
            plt.plot([i for i in range(1, len(d) + 1)], d.values(), color=self.colors[i], label=tags[i])
            i += 1
        plt.ylabel('std dwell time (seconds)')
        plt.xlabel('stop number')
        plt.legend()
        plt.savefig(paths[1])
        plt.close()
        i = 0
        for d in dt_tot_set:
            sns.kdeplot(d, label=tags[i], color=self.colors[i])
            i += 1
        plt.xlabel('total trip dwell time (seconds)')
        plt.legend()
        plt.savefig(paths[2])
        plt.close()
        return

    def trip_time_dist_validation(self):
        trip_time_set = []
        for trip in self.cp_trips:
            trip_time = tot_trip_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            trip_time_set.append(trip_time)
        trip_time_input = load('in/xtr/rt_20-2019-09/trip_times.pkl')
        trip_time_set.append(trip_time_input)
        plot_travel_time_benchmark(trip_time_set, ['simulated', 'observed'], self.colors,
                                   pathname='out/validation/ttd.png')
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

    def load_profile_base(self):
        lp, _, _, ons, offs = pax_per_trip_from_trajectory_set(self.cp_trips[0], IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
        through = np.subtract(lp, offs)
        through = through.tolist()
        plot_load_profile(ons, offs, lp, STOPS, through, pathname='out/load_profile_NC.png',
                          x_y_lbls=['stop id', 'number of passengers', 'number of through passengers and passenger load'],
                          controlled_stops=CONTROLLED_STOPS)
        return
