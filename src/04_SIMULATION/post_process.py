import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from copy import deepcopy


def write_trajectories(trip_data, pathname, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trip in trip_data:
            for s in trip_data[trip]:
                stop_lst = deepcopy(s)
                stop_lst.insert(0, trip)
                wf.writerow(stop_lst)
            i += 1
    return


def plot_headway(pathname, hs, ordered_stops, controlled_stops=None, min_size_for_cv=2, hw_scale=(200, 340, 20),
                 cv_scale=(0,1,0.1)):
    x = []
    y1 = []
    y2 = []
    for i in range(len(ordered_stops)):
        s = ordered_stops[i]
        if s in hs:
            h = hs[s]
            if len(h) > min_size_for_cv:
                mean = np.array(h).mean()
                std = np.array(h).std()
                cv = std/mean
                x.append(i)
                y1.append(cv) if mean else y1.append(0)
                y2.append(mean)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('stop id')
    ax1.set_ylabel('coefficient of variation', color=color)
    ax1.plot(x, y1, color=color)
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1]+cv_scale[2], cv_scale[2]))
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mean headway (seconds)', color=color)
    ax2.plot(x, y2, color=color)
    ax2.set_yticks(np.arange(hw_scale[0], hw_scale[1]+hw_scale[2], hw_scale[2]))
    ax2.tick_params(axis='y', labelcolor=color)

    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x1 = np.arange(len(ordered_stops))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ordered_stops, fontsize=6, rotation=90)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_trajectories(trip_data, idx_arr_t, idx_dep_t, pathname, ordered_stops, controlled_stops=None):
    for trip in trip_data:
        td = np.array(trip_data[trip])
        if np.size(td):
            arr_times = td[:, idx_arr_t].astype(float)
            dep_times = td[:, idx_dep_t].astype(float)
            times = np.vstack((arr_times, dep_times))
            times = times.flatten(order='F')
            starting_stop = td[0, 0]
            starting_stop_idx = ordered_stops.index(starting_stop)
            y_axis = np.arange(starting_stop_idx, starting_stop_idx + len(arr_times))
            y_axis = np.repeat(y_axis, 2)
            plt.plot(times, y_axis)
    if controlled_stops:
        for c in controlled_stops:
            stop_idx = ordered_stops.index(c)
            plt.axhline(y=stop_idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.yticks(np.arange(len(ordered_stops)), ordered_stops, fontsize=6)
    plt.xlabel('seconds')
    plt.ylabel('stops')
    plt.tick_params(labelright=True)
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_multiple_bar_charts(wta, wtc, pathname, lbls, ordered_stops, x_y_lbls=None):
    w = 0.27
    bar1 = np.arange(len(ordered_stops))
    bar2 = [i + w for i in bar1]
    x = ordered_stops
    y1 = []
    y2 = []
    for s in ordered_stops:
        y1.append(wta[s]) if s in wta else y1.append(0)
        y2.append(wtc[s]) if s in wtc else y2.append(0)
    plt.bar(bar1, y1, w, label=lbls[0], color='b')
    plt.bar(bar2, y2, w, label=lbls[1], color='r')
    plt.xticks(bar1, x, rotation=90, fontsize=6)
    if x_y_lbls:
        plt.xlabel(x_y_lbls[0])
        plt.ylabel(x_y_lbls[1])
    plt.tight_layout()
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_bar_chart(var, ordered_stops, pathname, x_y_lbls=None, controlled_stops=None):
    x = ordered_stops
    y1 = []
    for s in ordered_stops:
        y1.append(var[s]) if s in var else y1.append(0)
    plt.bar(x, y1)
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.xticks(rotation=90, fontsize=6)
    if x_y_lbls:
        plt.xlabel(x_y_lbls[0])
        plt.ylabel(x_y_lbls[1])
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def write_wait_times(mean_wait_time, stop_gps, pathname, ordered_stops):
    wait_times = pd.DataFrame(mean_wait_time.items(), columns=['stop', 'wait_time_sec'])
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    wait_times_df = pd.merge(wait_times, s, on='stop')

    stop_seq = [i for i in range(len(ordered_stops))]
    ordered_stop_data = {'stop': ordered_stops, 'stop_sequence': stop_seq}
    os_df = pd.DataFrame(ordered_stop_data)
    os_df['stop'] = os_df['stop'].astype(str)

    wait_times_df = pd.merge(wait_times_df, os_df, on='stop')
    wait_times_df = wait_times_df.sort_values(by=['stop_sequence'])
    wait_times_df.to_csv(pathname, index=False)
    return


def write_travel_times(pathname, link_times_mean, link_times_std, nr_time_dpoints):
    with open(pathname, 'w') as f:
        fw = csv.writer(f)
        for key in link_times_mean:
            fw.writerow([key])
            fw.writerow(nr_time_dpoints[key])
            fw.writerow(link_times_mean[key])
            fw.writerow(link_times_std[key])
    return


def plot_pax_per_stop(pathname, pax, ordered_stops, x_y_lbls, controlled_stops=None):
    fig, ax1 = plt.subplots()
    for stop in ordered_stops:
        if stop in pax:
            ax1.bar(stop, pax[stop], color='b')
        else:
            ax1.bar(stop, np.nan, color='b')
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x1 = np.arange(len(ordered_stops))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ordered_stops, fontsize=6, rotation=90)
    if x_y_lbls:
        plt.xlabel(x_y_lbls[0])
        plt.ylabel(x_y_lbls[1])
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_load_profile(bd, al, l, os, l_dev=None, pathname=None, x_y_lbls=None, controlled_stops=None):
    w = 0.3
    x1 = np.arange(len(os))
    x2 = [i + w for i in x1]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    y1, y2, y3, y4 = [], [], [], []
    for stop in os:
        if stop in bd:
            y1.append(bd[stop])
        else:
            y1.append(0)
        if stop in al:
            y2.append(al[stop])
        else:
            y2.append(0)
        if stop in l:
            y3.append(l[stop])
        else:
            y3.append(0)
        if l_dev:
            if stop in l_dev:
                y4.append(l_dev[stop])
            else:
                y4.append(0)

    ax1.bar(x1, y1, w, label='ons', color='g')
    ax1.bar(x2, y2, w, label='offs', color='r')
    ax2.plot(x1, y3, label='load', color='dodgerblue')
    if l_dev:
        ax2.plot(x1, y4, label='load deviation', color='purple')
    if controlled_stops:
        for cs in controlled_stops:
            idx = os.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(os, fontsize=6, rotation=90)
    # right, left, top, bottom
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1], color='black')
        ax2.set_ylabel(x_y_lbls[2], color='dodgerblue')
        ax2.tick_params(axis='y', colors='dodgerblue')
    plt.tight_layout()
    fig.legend(loc='upper center')
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def get_headway_from_trajectory_set(trajectory_set, idx_ons, idx_denied, first_trip):
    recorded_headway = {}
    tot_wait_time = {}
    wait_time = {}
    wait_time_from_hw = {}
    tot_boardings = {}
    for trajectories in trajectory_set:
        prev_stop_time = {}
        prev_denied = {}
        for trip in trajectories:
            if trip != first_trip:
                for stop_details in trajectories[trip]:
                    stop_id = stop_details[0]
                    stop_time = stop_details[1]
                    ons = stop_details[idx_ons]
                    denied = stop_details[idx_denied]
                    if stop_id not in prev_stop_time:
                        prev_stop_time[stop_id] = stop_time
                        prev_denied[stop_id] = denied
                    else:
                        t1 = prev_stop_time[stop_id]
                        t2 = stop_time
                        headway = t2 - t1
                        prev_denied_ = prev_denied[stop_id]
                        if stop_id not in recorded_headway:
                            recorded_headway[stop_id] = [headway]
                            tot_wait_time[stop_id] = (prev_denied_ + (ons+denied-prev_denied_) * 0.5) * headway
                            tot_boardings[stop_id] = ons
                        else:
                            recorded_headway[stop_id].append(headway)
                            tot_wait_time[stop_id] += (prev_denied_ + (ons+denied-prev_denied_) * 0.5) * headway
                            tot_boardings[stop_id] += ons

                        prev_stop_time[stop_id] = t2
                        prev_denied[stop_id] = denied
    for s in tot_wait_time:
        wait_time[s] = tot_wait_time[s] / tot_boardings[s] if tot_boardings[s] else 0
        headway = np.array(recorded_headway[s])
        mean_headway = headway.mean()
        cv_headway = headway.std() / mean_headway
        wait_time_from_hw[s] = (mean_headway / 2) * (1 + (cv_headway * cv_headway))

    return recorded_headway, wait_time, wait_time_from_hw


def pax_per_trip_from_trajectory_set(trajectory_set, idx_load, idx_ons, idx_offs, first_trip):
    bus_load_all = {}
    ons_all = {}
    offs_all = {}
    for trajectories in trajectory_set:
        for trip in trajectories:
            if trip != first_trip:
                for stop_details in trajectories[trip]:
                    stop_id = stop_details[0]
                    bus_load = stop_details[idx_load]
                    ons = stop_details[idx_ons]
                    offs = stop_details[idx_offs]

                    if stop_id not in bus_load_all:
                        bus_load_all[stop_id] = [bus_load]
                    else:
                        bus_load_all[stop_id].append(bus_load)

                    if stop_id not in ons_all:
                        ons_all[stop_id] = [ons]
                    else:
                        ons_all[stop_id].append(ons)

                    if stop_id not in offs_all:
                        offs_all[stop_id] = [offs]
                    else:
                        offs_all[stop_id].append(offs)

    bus_load_mean = {}
    bus_load_std = {}
    ons_mean = {}
    ons_std = {}
    offs_mean = {}
    offs_std = {}
    for stop in bus_load_all:
        bus_load_mean[stop] = np.nan_to_num(np.array(bus_load_all[stop]).mean())
        bus_load_std[stop] = np.nan_to_num(np.array(bus_load_all[stop]).std())

    for stop in ons_all:
        ons_mean[stop] = np.nan_to_num(np.array(ons_all[stop]).mean())
        # ons_std[stop] = np.nan_to_num(np.array(ons_all[stop]).std())

    for stop in offs_all:
        offs_mean[stop] = np.nan_to_num(np.array(offs_all[stop]).mean())
        # offs_std[stop] = np.nan_to_num(np.array(offs_all[stop]).std())

    return bus_load_mean, bus_load_std, ons_mean, offs_mean


def hold_time_from_trajectory_set(trajectory_set, idx, first_trip, controlled_stops):
    ht_all_in_one = []
    ht_all = {}
    for trajectories in trajectory_set:
        for trip in trajectories:
            if trip != first_trip:
                for stop_details in trajectories[trip]:
                    stop_id = stop_details[0]
                    ht = stop_details[idx]
                    if stop_id in controlled_stops:
                        ht_all_in_one.append(ht)
                    if stop_id not in ht_all:
                        ht_all[stop_id] = [ht]
                    else:
                        ht_all[stop_id].append(ht)
    ht_mean = {}
    for stop in ht_all:
        ht_mean[stop] = np.nan_to_num(np.array(ht_all[stop]).mean())
    return ht_mean, ht_all_in_one


def denied_from_trajectory_set(trajectory_set, idx, tot_ons, first_trip):
    tot_denied = {}
    for trajectories in trajectory_set:
        for trip in trajectories:
            if trip != first_trip:
                for stop_details in trajectories[trip]:
                    stop_id = stop_details[0]
                    denied = stop_details[idx]
                    if stop_id not in tot_denied:
                        tot_denied[stop_id] = denied
                    else:
                        tot_denied[stop_id] += denied
    per_mil_denied = {}
    for stop in tot_denied:
        if tot_ons:
            per_mil_denied[stop] = tot_denied[stop] / tot_ons[stop] * 1000
    return per_mil_denied


def travel_times_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t, first_trip):
    link_times = {}
    dwell_times = {}
    for trajectory in trajectory_set:
        for trip in trajectory:
            if trip != first_trip:
                trip_data = trajectory[trip]
                for i in range(len(trip_data) - 1):
                    s0 = trip_data[i][0]
                    s1 = trip_data[i+1][0]
                    link = s0 + '-' + s1
                    link_time = trip_data[i + 1][idx_arr_t] - trip_data[i][idx_dep_t]
                    if link in link_times:
                        link_times[link].append(link_time)
                    else:
                        link_times[link] = [link_time]
                    dwell_time = trip_data[i][idx_dep_t] - trip_data[i][idx_arr_t]
                    if s0 in dwell_times:
                        dwell_times[s0].append(dwell_time)
                    else:
                        dwell_times[s0] = [dwell_time]
    mean_link_times = {}
    std_link_times = {}
    mean_dwell_times = {}
    std_dwell_times = {}
    for link in link_times:
        mean_link_times[link] = round(np.array(link_times[link]).mean(), 1)
        std_link_times[link] = round(np.array(link_times[link]).std(), 1)
    for s in dwell_times:
        mean_dwell_times[s] = round(np.array(dwell_times[s]).mean(), 1)
        std_dwell_times[s] = round(np.array(dwell_times[s]).std(), 1)
    return mean_link_times, std_link_times, mean_dwell_times, std_dwell_times


def plot_link_times(link_times_mean, link_times_std, ordered_stops, pathname, lbls, x_y_lbls=None, controlled_stops=None):
    w = 0.27
    bar1 = np.arange(len(ordered_stops)-1)
    bar2 = [i + w for i in bar1]
    mean = []
    std = []
    links = [i + '-' + j for i, j in zip(ordered_stops[:-1], ordered_stops[1:])]
    for link in links:
        mean.append(link_times_mean[link]) if link in link_times_mean else mean.append(0)
        std.append(link_times_std[link]) if link in link_times_std else std.append(0)
    plt.bar(bar1, mean, w, label=lbls[0], color='b')
    plt.bar(bar2, std, w, label=lbls[1], color='r')
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.xticks(bar1, links, rotation=90, fontsize=6)
    if x_y_lbls:
        plt.xlabel(x_y_lbls[0])
        plt.ylabel(x_y_lbls[1])
    plt.tight_layout()
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def write_link_times(link_times_mean, link_times_std, stop_gps, pathname, ordered_stops):

    link_times_df = pd.DataFrame(link_times_mean.items(), columns=['stop_1', 'mean_sec'])
    link_times_df['std_sec'] = link_times_df['stop_1'].map(link_times_std)
    link_times_df[['stop_1', 'stop_2']] = link_times_df['stop_1'].str.split('-', expand=True)
    link_times_df = link_times_df[['stop_1', 'stop_2', 'mean_sec', 'std_sec']]

    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)

    s1 = s.rename(columns={'stop_id': 'stop_1', 'stop_lat': 'stop_1_lat', 'stop_lon': 'stop_1_lon'})
    link_times_df = pd.merge(link_times_df, s1, on='stop_1')
    s2 = s1.rename(columns={'stop_1': 'stop_2', 'stop_1_lat': 'stop_2_lat', 'stop_1_lon': 'stop_2_lon'})
    link_times_df = pd.merge(link_times_df, s2, on='stop_2')

    stop_seq = [i for i in range(len(ordered_stops)-1)]
    ordered_stop_data = {'stop_1': ordered_stops[:-1], 'stop_1_sequence': stop_seq}
    os_df = pd.DataFrame(ordered_stop_data)
    os_df['stop_1'] = os_df['stop_1'].astype(str)
    link_times_df = pd.merge(link_times_df, os_df, on='stop_1')
    link_times_df = link_times_df.sort_values(by=['stop_1_sequence'])

    link_times_df.to_csv(pathname, index=False)
    return


def write_dwell_times(dwell_times_mean, dwell_times_std, stop_gps, pathname, ordered_stops):
    dwell_times_std_df = pd.DataFrame(dwell_times_std.items(), columns=['stop', 'std_sec'])
    dwell_times_mean_df = pd.DataFrame(dwell_times_mean.items(), columns=['stop', 'mean_sec'])
    dwell_times_df = pd.merge(dwell_times_mean_df, dwell_times_std_df, on='stop')
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    dwell_times_df = pd.merge(dwell_times_df, s, on='stop')

    stop_seq = [i for i in range(len(ordered_stops))]
    ordered_stop_data = {'stop': ordered_stops, 'stop_sequence': stop_seq}
    os_df = pd.DataFrame(ordered_stop_data)
    os_df['stop'] = os_df['stop'].astype(str)

    dwell_times_df = pd.merge(dwell_times_df, os_df, on='stop')
    dwell_times_df = dwell_times_df.sort_values(by=['stop_sequence'])
    dwell_times_df.to_csv(pathname, index=False)
    return


def plot_dwell_times(dwell_times_mean, dwell_times_std, ordered_stops, pathname, lbls, x_y_lbls=None, controlled_stops=None):
    w = 0.27
    bar1 = np.arange(len(ordered_stops))
    bar2 = [i + w for i in bar1]
    mean = []
    std = []
    x = ordered_stops
    for s in ordered_stops:
        mean.append(dwell_times_mean[s]) if s in dwell_times_mean else mean.append(0)
        std.append(dwell_times_std[s]) if s in dwell_times_std else std.append(0)
    plt.bar(bar1, mean, w, label=lbls[0], color='b')
    plt.bar(bar2, std, w, label=lbls[1], color='r')
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.xticks(bar1, x, rotation=90, fontsize=6)
    if x_y_lbls:
        plt.xlabel(x_y_lbls[0])
        plt.ylabel(x_y_lbls[1])
    plt.tight_layout()
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_histogram(data, pathname):
    plt.hist(data, density=True)
    plt.xlabel('seconds')
    plt.ylabel('frequency')
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def get_stop_loc(pathname):
    stop_gps = pd.read_csv(pathname)
    stop_gps = stop_gps[['stop_id', 'stop_lat', 'stop_lon']]
    return stop_gps


def save(pathname, par):
    with open(pathname, 'wb') as tf:
        pickle.dump(par, tf)
    return


def load(pathname):
    with open(pathname, 'rb') as tf:
        var = pickle.load(tf)
    return var


def get_historical_headway(pathname, dates, all_stops, trips, start_time, end_time, early_departure=0.5*60,
                           sch_dev_tolerance=6*60):
    whole_df = pd.read_csv(pathname)
    df_period = whole_df[whole_df['trip_id'].isin(trips)]
    headway = {}
    for d in dates:
        df_temp = df_period[df_period['event_time'].astype(str).str[:10] == d]
        df_temp = df_temp[df_temp['avl_sec'] % 86400 <= end_time]
        df_temp = df_temp[df_temp['avl_sec'] % 86400 >= start_time]
        for s in all_stops:
            df_stop = df_temp[df_temp['stop_id'] == int(s)]
            for i, j in zip(trips, trips[1:]):
                t2_df = df_stop[df_stop['trip_id'] == j]
                t1_df = df_stop[df_stop['trip_id'] == i]
                if (not t1_df.empty) & (not t2_df.empty):
                    t1_avl = float(t1_df['avl_sec']) % 86400
                    t2_avl = float(t2_df['avl_sec']) % 86400
                    t1_schd = float(t1_df['schd_sec'])
                    t2_schd = float(t2_df['schd_sec'])
                    condition1 = s == all_stops[0]
                    condition2 = t1_schd - t1_avl > early_departure or t2_schd - t2_avl > early_departure
                    sch_dev_condition = abs(t1_schd - t1_avl) > sch_dev_tolerance or abs(t2_schd-t2_avl) > sch_dev_tolerance
                    faulty = condition1 and condition2

                    if not faulty and not sch_dev_condition:
                        hw = t2_avl - t1_avl
                        if s in headway:
                            headway[s].append(hw)
                        else:
                            headway[s] = [hw]
    return headway


def get_input_boardings(arrival_rates, dem_interval_len_min, start_time_sec, end_time_sec, first_interval):
    aggregated_boardings = {}
    start_interval = int(start_time_sec / (60*dem_interval_len_min))
    end_interval = int(end_time_sec / (60*dem_interval_len_min))
    start_idx = start_interval - first_interval
    end_idx = end_interval - first_interval
    for s in arrival_rates:
        arr = arrival_rates[s]
        agg = sum([a*dem_interval_len_min for a in arr[start_idx:end_idx]])
        aggregated_boardings[s] = agg
    return aggregated_boardings



