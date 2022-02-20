import matplotlib.pyplot as plt
import numpy as np

from input import *


class PostProcessor:
    def __init__(self, cp_trip_paths, cp_pax_paths, cp_tags):
        self.colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        self.cp_trips, self.cp_pax, self.cp_tags = [], [], []
        for trip_path, pax_path, tag in zip(cp_trip_paths, cp_pax_paths, cp_tags):
            self.cp_trips.append(load(trip_path))
            self.cp_pax.append(load(pax_path))
            self.cp_tags.append(tag)

    def pax_times(self):
        jt_mean = []
        rbt_mean = []
        wt_mean = []
        rt_mean = []
        pct_ewt_mean = []
        db_wt_all = []
        for pax in self.cp_pax:
            jt, rbt, wt, rt, pct_ewt, db_wt = get_pax_times(pax, STOPS, FOCUS_TRIPS_MEAN_HW/2)
            jt_mean.append(jt)
            rbt_mean.append(rbt)
            wt_mean.append(wt)
            rt_mean.append(rt)
            pct_ewt_mean.append(pct_ewt)
            db_wt_all.append(db_wt)
        print(jt_mean)
        print(rbt_mean)
        print(wt_mean)
        print(rt_mean)
        print(pct_ewt_mean)
        print(db_wt_all)
        return

    def headway(self):
        cv_hw_set = []
        for trips in self.cp_trips:
            temp_cv_hw, hw_at_tp = get_headway_from_trajectory_set(trips, IDX_ARR_T, STOPS,
                                                                   controlled_stops=CONTROLLED_STOPS)
            cv_hw_set.append(temp_cv_hw)
        plot_headway_benchmark(cv_hw_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/hw.png',
                               controlled_stops=CONTROLLED_STOPS[:-1])

        print(f'mean cv headway {[np.mean(cv) for cv in cv_hw_set]}')
        return

    def total_trip_time_distribution(self):
        trip_time_set = []
        extreme_trip_time_set = []
        for trip in self.cp_trips:
            trip_time, extreme_trip_time = tot_trip_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            trip_time_set.append(trip_time)
            extreme_trip_time_set.append(extreme_trip_time)
        plot_travel_time_benchmark(trip_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ttd.png')
        print(f'trip time 95th percentile {extreme_trip_time_set}')
        return

    def load_profile(self):
        load_profile_set = []
        lp_std_set = []
        for trips in self.cp_trips:
            temp_lp, temp_lp_std, _, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            load_profile_set.append(temp_lp)
            lp_std_set.append(temp_lp_std)
        plot_load_profile_benchmark(load_profile_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/lp.png',
                                    controlled_stops=CONTROLLED_STOPS, x_y_lbls=['stop id', 'avg load per trip'])
        plot_load_profile_benchmark(lp_std_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/lp_sd.png',
                                    controlled_stops=CONTROLLED_STOPS, x_y_lbls=['stop id', 'sd load per trip'])
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

    def hold_time(self):
        hold_time_set = []
        ht_per_stop_set = []
        for trips in self.cp_trips:
            tot_ht_mean, ht_per_stop = hold_time_from_trajectory_set(trips, IDX_HOLD_TIME, 0, CONTROLLED_STOPS)
            hold_time_set.append(tot_ht_mean)
            ht_per_stop_set.append(ht_per_stop)
        plot_mean_hold_time_benchmark(hold_time_set, self.cp_tags, self.colors, pathname='out/benchmark/ht.png')
        print(f'hold time mean {hold_time_set}')
        print(f'control stops {CONTROLLED_STOPS}')
        print(f'ht mean per control stop {ht_per_stop_set}')
        return

    def denied(self):
        denied_set = []
        for trips in self.cp_trips:
            _, _, tot_ons, _, _ = pax_per_trip_from_trajectory_set(trips, IDX_LOAD, IDX_PICK, IDX_DROP, STOPS)
            denied_set.append(denied_from_trajectory_set(trips, IDX_DENIED, tot_ons))
        plot_denied_benchmark(denied_set, self.cp_tags, self.colors, pathname='out/benchmark/db.png')
        return

    def wait_times_per_stop(self):
        wait_time_set = []
        counter = 0
        for pax in self.cp_pax:
            wt_mean = get_wait_times(pax, STOPS)
            wait_time_set.append(wt_mean)
            counter += 1
        scheduled_avg_wait = FOCUS_TRIPS_MEAN_HW / 2
        plot_wait_time_benchmark(wait_time_set, STOPS, self.cp_tags, self.colors, pathname='out/benchmark/wt.png',
                                 scheduled_wait=scheduled_avg_wait, controlled_stops=CONTROLLED_STOPS)
        return

    def write_trajectories(self):
        i = 0
        for trips in self.cp_trips:
            write_trajectory_set(trips, 'out/trajectories_' + self.cp_tags[i] + '.csv', IDX_ARR_T, IDX_DEP_T, IDX_HOLD_TIME,
                                 header=['trip_id', 'stop_id', 'arr_t', 'dep_t', 'pax_load', 'ons', 'offs', 'denied',
                                         'hold_time', 'skipped', 'replication', 'arr_sec', 'dep_sec', 'dwell_sec'])
            i += 1
        return

    def dwell_time_validation(self):
        dt_tot_set = []
        for trip in self.cp_trips:
            _, _, _, _, dt_tot = travel_times_from_trajectory_set(trip, IDX_DEP_T, IDX_ARR_T)
            dt_tot_set.append(dt_tot)
            print(f'output mean {np.mean(dt_tot)}')
            print(f'output std {np.std(dt_tot)}')
        paths = ('out/benchmark/dwell_t_mean.png', 'out/benchmark/dwell_t_std.png', 'out/benchmark/dwell_t_tot.png')
        dt_input_tot = load('in/xtr/rt_20-2019-09/dwell_times_tot.pkl')
        print(f'input mean {np.mean(dt_input_tot)}')
        print(f'input std {np.std(dt_input_tot)}')
        dt_tot_set.append(dt_input_tot)
        tags = ['simulated', 'observed']
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
            print(f'output mean {np.mean(trip_time)}')
            print(f'output std {np.std(trip_time)}')
        trip_time_input = load('in/xtr/rt_20-2019-09/trip_times.pkl')
        print(f'input mean {np.mean(trip_time_input)}')
        print(f'input std {np.std(trip_time_input)}')
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
                          x_y_lbls=['stop', 'passengers (per trip)', 'through passengers and passenger load (per trip)'],
                          controlled_stops=CONTROLLED_STOPS[:-1])
        return


def dwell_times(stops, df_nc, df_eh, df_rl, nr_replications, header):
    dwell_df_rows = []
    for s in stops:
        dwell_df_row = [int(s)]
        for df in (df_nc, df_eh, df_rl):
            dwell_time_means = []
            dwell_times_sd = []
            for n in range(1, nr_replications + 1):
                rep_df = df[df['replication'] == n]
                stop_dwell_times = rep_df[rep_df['stop_id'] == int(s)]['dwell_sec'].to_numpy()
                dwell_time_means.append(stop_dwell_times.mean())
                dwell_times_sd.append(stop_dwell_times.std())
            dwell_df_row.append(np.around(np.mean(dwell_time_means), decimals=1))
            dwell_df_row.append(np.around(np.mean(dwell_times_sd), decimals=1))
        dwell_df_rows.append(dwell_df_row)
    dwell_df = pd.DataFrame(dwell_df_rows, columns=header)
    dwell_df.to_csv('out/dwell_times.csv', index=False)


def link_times(links, df_nc, df_eh, df_rl, nr_replications, header):
    link_times_df_rows = []
    for li in links:
        link_df_row = [str(li[0]) + '-' + str(li[1])]
        for df in (df_nc, df_eh, df_rl):
            link_time_means = []
            link_times_sd = []
            for n in range(1, nr_replications + 1):
                rep_df = df[df['replication'] == n]
                dep_sec = rep_df[rep_df['stop_id'] == int(li[0])]['dep_sec'].values
                arr_sec = rep_df[rep_df['stop_id'] == int(li[1])]['arr_sec'].values
                li_times = arr_sec - dep_sec
                link_time_means.append(li_times.mean())
                link_times_sd.append(li_times.std())
            link_df_row.append(np.around(np.mean(link_time_means), decimals=1))
            link_df_row.append(np.around(np.mean(link_times_sd), decimals=1))
        link_times_df_rows.append(link_df_row)
    link_times_df = pd.DataFrame(link_times_df_rows, columns=header)
    link_times_df.to_csv('out/link_times.csv', index=False)


def headway(stops, df_nc, df_eh, df_rl, nr_replications, header):
    headway_df_rows = []
    for s in stops:
        hw_df_row = [int(s)]
        for df in (df_nc, df_eh, df_rl):
            headway_means = []
            headway_sd = []
            for n in range(1, nr_replications + 1):
                rep_df = df[df['replication'] == n]
                stop_times = rep_df[rep_df['stop_id'] == int(s)]['arr_sec'].to_list()
                stop_hws = [i - j for i, j in zip(stop_times[1:], stop_times[:-1])]
                headway_means.append(np.around(np.mean(stop_hws), decimals=1))
                headway_sd.append(np.around(np.std(stop_hws), decimals=1))
            hw_df_row.append(np.around(np.mean(headway_means), decimals=1))
            hw_df_row.append(np.around(np.mean(headway_sd), decimals=1))
        headway_df_rows.append(hw_df_row)
    headway_df = pd.DataFrame(headway_df_rows, columns=header)
    headway_df.to_csv('out/headway.csv', index=False)


def error_headway(stops, df_nc, df_eh, df_rl, nr_replications):
    errors = []
    for df in (df_nc, df_eh, df_rl):
        sd_hw_reps = []
        for n in range(1, nr_replications + 1):
            all_hw = []
            rep_df = df[df['replication'] == n]
            for s in stops:
                stop_times = rep_df[rep_df['stop_id'] == int(s)]['arr_sec'].to_list()
                stop_hws = [i - j for i, j in zip(stop_times[1:], stop_times[:-1])]
                all_hw += stop_hws
            sd_hw_reps.append(np.std(all_hw))
        print(sd_hw_reps)
        errors.append(100 * np.std(sd_hw_reps) / np.mean(sd_hw_reps))
    print(errors)
