from input import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)

    def headway(self):
        hw_set = []
        sd_hw_at_tp = []
        for trips in self.cp_trips:
            temp_hw, _, hw_at_tp = get_headway_from_trajectory_set(trips, IDX_PICK, IDX_DENIED, IDX_ARR_T,
                                                                   controlled_stops=CONTROLLED_STOPS)
            hw_set.append(temp_hw)

            sd_hw_at_tp.append(np.array(hw_at_tp).std())
        plot_headway_benchmark(hw_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/hw.png',
                               controlled_stops=CONTROLLED_STOPS)
        # print(sd_hw_at_tp)
        return

    def total_trip_time_distribution(self):
        trip_time_set = []
        for trip in self.cp_trips:
            trip_time = tot_trip_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            trip_time_set.append(trip_time)
        std_run_times = []
        extr_run_times = []
        for t in trip_time_set:
            t_arr = np.array(t)
            std_run_times.append(t_arr.std())
            extr_run_times.append(np.percentile(t_arr, 95))
        # print(std_run_times)
        # print(extr_run_times)
        plot_travel_time_benchmark(trip_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ttd.png')
        return

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, _, _, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP)
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
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
        plot_mean_hold_time_benchmark(hold_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ht.png')
        plot_hold_time_distribution_benchmark(hold_time_dist_set, self.cp_tags, self.colors,
                                              pathname='out/benchmark/ht_dist.png')
        return

    def denied(self):
        denied_set = []
        for trips in self.cp_trips:
            _, _, _, _, _, tot_ons = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP)
            denied_set.append(denied_from_trajectory_set(trips, IDX_DENIED, tot_ons))
        plot_denied_benchmark(denied_set, self.cp_tags, self.colors, pathname='out/benchmark/db.png')
        return

    def wait_times(self):
        wait_time_set = []
        percentile_wt_set = []
        for pax in self.cp_pax:
            wt_mean, wt_std, percentile_wt = get_wait_times(pax, STOPS)
            wait_time_set.append(wt_mean)
            percentile_wt_set.append(percentile_wt)
        plot_wait_time_benchmark(wait_time_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/wt.png')
        return

    def rbt_difference(self, a, b):
        idx1 = self.cp_tags.index(a)
        idx2 = self.cp_tags.index(b)
        _, _, rbt1, _, _, _ = get_journey_times(self.cp_pax[idx1], STOPS)
        _, _, rbt2, _, _, _ = get_journey_times(self.cp_pax[idx2], STOPS)
        plot_difference_od(rbt1-rbt2, STOPS, controlled_stops=CONTROLLED_STOPS,
                           pathname='out/benchmark/rbt_diff_'+self.cp_tags[idx1] + '_' + self.cp_tags[idx2] + '.png',
                           clim=(-250, 250))
        return

    def journey_times(self):
        jt_sums = []
        extr_jt_sums = []
        for pax in self.cp_pax:
            _, _, _, _, jt_sum, extr_jt_sum = get_journey_times(pax, STOPS)
            jt_sums.append(jt_sum)
            extr_jt_sums.append(extr_jt_sum)
        # print(jt_sums)
        # print(extr_jt_sums)
        return

    def params_for_policy(self, tag='RL'):
        idx = self.cp_tags.index(tag)
        mean_load_comb, _, ons_comb, _, _, _ = pax_per_trip_from_trajectory_set(self.cp_trips[idx], IDX_LOAD,
                                                                                IDX_PICK, IDX_DROP)
        return mean_load_comb, ons_comb