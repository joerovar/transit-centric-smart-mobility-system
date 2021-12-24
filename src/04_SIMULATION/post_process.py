import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from copy import deepcopy
import seaborn as sns
from datetime import timedelta
import matplotlib


def write_trajectories(trip_data, pathname, idx_arr_t, idx_dep_t, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trip in trip_data:
            for s in trip_data[trip]:
                stop_lst = deepcopy(s)
                stop_lst[idx_arr_t] = str(timedelta(seconds=round(stop_lst[idx_arr_t])))
                stop_lst[idx_dep_t] = str(timedelta(seconds=round(stop_lst[idx_dep_t])))
                stop_lst.insert(0, trip)
                wf.writerow(stop_lst)
            i += 1
    return


def write_sars(trip_data, pathname, header=None):
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


def plot_multiple_bar_charts(wta, wtc,  lbls, ordered_stops, x_y_lbls=None, pathname=None):
    w = 0.5
    bar1 = np.arange(len(ordered_stops))
    bar2 = [i + w for i in bar1]
    x = ordered_stops
    y1 = []
    y2 = []
    for s in ordered_stops:
        y1.append(wta[s]) if s in wta else y1.append(0)
        y2.append(wtc[s]) if s in wtc else y2.append(0)
    plt.bar(bar1, y1, w, label=lbls[0], color='tab:blue')
    plt.bar(bar2, y2, w, label=lbls[1], color='tab:red')
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


def get_headway_from_trajectory_set(trajectory_set, idx_ons, idx_denied, idx_arr_t, controlled_stops=None):
    recorded_headway = {}
    tot_wait_time = {}
    wait_time = {}
    wait_time_from_hw = {}
    tot_boardings = {}
    hw_at_tp = []
    for trajectories in trajectory_set:
        prev_stop_time = {}
        prev_denied = {}
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                stop_time = stop_details[idx_arr_t]
                ons = stop_details[idx_ons]
                denied = stop_details[idx_denied]
                if stop_id not in prev_stop_time:
                    prev_stop_time[stop_id] = stop_time
                    prev_denied[stop_id] = denied
                else:
                    t1 = prev_stop_time[stop_id]
                    t2 = stop_time
                    headway = t2 - t1
                    if stop_id in controlled_stops:
                        hw_at_tp.append(headway)
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

    return recorded_headway, wait_time, hw_at_tp


def pax_per_trip_from_trajectory_set(trajectory_set, idx_load, idx_ons, idx_offs):
    bus_load_all = {}
    ons_all = {}
    offs_all = {}
    total_ons = 0
    for trajectories in trajectory_set:
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                bus_load = stop_details[idx_load]
                ons = stop_details[idx_ons]
                total_ons += ons
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
    offs_mean = {}
    ons_tot = {}

    for stop in bus_load_all:
        bus_load_mean[stop] = np.nan_to_num(np.array(bus_load_all[stop]).mean())
        bus_load_std[stop] = np.nan_to_num(np.array(bus_load_all[stop]).std())

    for stop in ons_all:
        ons_mean[stop] = np.nan_to_num(np.array(ons_all[stop]).mean())
        ons_tot[stop] = np.nan_to_num(np.array(ons_all[stop]).sum())

    for stop in offs_all:
        offs_mean[stop] = np.nan_to_num(np.array(offs_all[stop]).mean())

    return bus_load_mean, bus_load_std, ons_mean, offs_mean, ons_tot, total_ons


def hold_time_from_trajectory_set(trajectory_set, idx):
    tot_hold_times = []
    ht_all = {}
    for trajectories in trajectory_set:
        for trip in trajectories:
            trip_hold_time = 0
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                ht = stop_details[idx]
                trip_hold_time += ht
                if stop_id not in ht_all:
                    ht_all[stop_id] = [ht]
                else:
                    ht_all[stop_id].append(ht)
            tot_hold_times.append(trip_hold_time)
    avg_tot_ht = np.array(tot_hold_times).mean()
    ht_mean = {}
    for stop in ht_all:
        ht_mean[stop] = np.nan_to_num(np.array(ht_all[stop]).mean())
    return ht_mean, avg_tot_ht, tot_hold_times


def denied_from_trajectory_set(trajectory_set, idx, tot_ons):
    tot_denied = 0
    for trajectories in trajectory_set:
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                denied = stop_details[idx]
                tot_denied += denied
    per_mil_denied = tot_denied * 1000 / tot_ons
    return per_mil_denied


def tot_trip_times_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t):
    tot_trip_times = []
    for trajectories in trajectory_set:
        lst_trips = list(trajectories.keys())
        trip_times = [trajectories[k][-1][idx_arr_t] - trajectories[k][0][idx_dep_t] for k in lst_trips]
        tot_trip_times += trip_times
    return tot_trip_times


def travel_times_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t):
    link_times = {}
    dwell_times = {}
    dwell_times_tot = []
    for trajectory in trajectory_set:
        for trip in trajectory:
            trip_data = trajectory[trip]
            temp_dwell_t = 0
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
                temp_dwell_t += dwell_time
                if i:
                    if s0 in dwell_times:
                        dwell_times[s0].append(dwell_time)
                    else:
                        dwell_times[s0] = [dwell_time]
            dwell_times_tot.append(temp_dwell_t)
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
    return mean_link_times, std_link_times, mean_dwell_times, std_dwell_times, dwell_times_tot


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
    plt.hist(data, ec='black')
    plt.xlabel('seconds')
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


def get_historical_headway(pathname, dates, all_stops, trips, early_departure=0.5*60,
                           sch_dev_tolerance=6*60):
    whole_df = pd.read_csv(pathname)
    df_period = whole_df[whole_df['trip_id'].isin(trips)]
    headway = {}
    for d in dates:
        df_temp = df_period[df_period['event_time'].astype(str).str[:10] == d]
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
                    # condition1 = s == all_stops[0]
                    # condition2 = t1_schd - t1_avl > early_departure or t2_schd - t2_avl > early_departure
                    # sch_dev_condition = abs(t1_schd - t1_avl) > sch_dev_tolerance or abs(t2_schd-t2_avl) > sch_dev_tolerance
                    # faulty = condition1 and condition2
                    #
                    # if not faulty and not sch_dev_condition:
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


def plot_od(od, ordered_stops, pathname=None, clim=None, controlled_stops=None):
    plt.imshow(od, interpolation='nearest')
    current_cmap = matplotlib.cm.get_cmap().copy()
    current_cmap.set_bad(color='white')
    if clim:
        plt.clim(clim)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('seconds', rotation=270)
    bar = np.arange(len(ordered_stops))
    x = np.array(ordered_stops)
    plt.xticks(bar, x, rotation=90, fontsize=6)
    plt.yticks(bar, x, fontsize=6)
    if controlled_stops:
        for c in controlled_stops:
            idx = ordered_stops.index(c)
            plt.axhline(y=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_difference_od(od, ordered_stops, pathname=None, clim=None, controlled_stops=None):
    plt.imshow(od, interpolation='nearest', cmap='coolwarm')
    current_cmap = matplotlib.cm.get_cmap().copy()
    current_cmap.set_bad(color='white')
    if clim:
        plt.clim(clim)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('seconds', rotation=270)
    bar = np.arange(len(ordered_stops))
    x = np.array(ordered_stops)
    plt.xticks(bar, x, rotation=90, fontsize=6)
    plt.yticks(bar, x, fontsize=6)
    for c in controlled_stops:
        idx = ordered_stops.index(c)
        plt.axhline(y=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def get_wait_times(pax_set, ordered_stops):
    wait_times = {'o': [], 'd': [], 'wt': []}
    for replication in pax_set:
        for p in replication:
            wait_times['o'].append(p.orig_idx)
            wait_times['d'].append(p.dest_idx)
            wait_times['wt'].append(p.wait_time)
    wait_times_df = pd.DataFrame(wait_times)
    n = len(ordered_stops)
    stop_wait_time_mean = np.zeros(n)
    stop_wait_time_mean[:] = np.nan
    stop_wait_time_std = np.zeros(n)
    stop_wait_time_std[:] = np.nan
    for i in range(n):
        wt = wait_times_df[(wait_times_df['o'] == i)]['wt']
        if not wt.empty:
            stop_wait_time_mean[i] = wt.mean()
            stop_wait_time_std[i] = wt.std()
    stop_nr = ordered_stops.index('443')
    point_wt = wait_times_df[(wait_times_df['o'] == stop_nr)]['wt']
    point_wt = point_wt.quantile(0.8)
    return stop_wait_time_mean, stop_wait_time_std, point_wt


def get_journey_times(pax_set, ordered_stops):
    # we add all data points to a dataframe
    # then we convert into an od matrix
    journey_times = {'o': [], 'd': [], 'jt': []}
    for replication in pax_set:
        for p in replication:
            journey_times['o'].append(p.orig_idx)
            journey_times['d'].append(p.dest_idx)
            journey_times['jt'].append(p.journey_time)
    journey_times_df = pd.DataFrame(journey_times)
    n = len(ordered_stops)
    od_journey_time_mean = np.zeros(shape=(n,)*2)
    od_journey_time_mean[:] = np.nan
    od_journey_time_std = np.zeros(shape=(n,)*2)
    od_journey_time_std[:] = np.nan
    od_journey_time_rbt = np.zeros(shape=(n,)*2)
    od_journey_time_rbt[:] = np.nan
    od_extr_journey_time_mean = np.zeros(shape=(n,)*2)
    od_extr_journey_time_mean[:] = np.nan
    od_journey_time_sum = np.zeros(shape=(n,)*2)
    od_journey_time_sum[:] = np.nan
    od_count = np.zeros(shape=(n,)*2)
    for i in range(n):
        for j in range(i + 1, n):
            jt = journey_times_df[(journey_times_df['o'] == i) & (journey_times_df['d'] == j)]['jt']
            if not jt.empty:
                od_count[i, j] = len(jt)
                od_journey_time_mean[i, j] = jt.mean()
                od_journey_time_std[i, j] = jt.std()
                od_journey_time_rbt[i, j] = jt.quantile(0.8) - jt.median()
                od_extr_journey_time_mean[i, j] = jt.quantile(0.95)
                od_journey_time_sum[i, j] = jt.sum()
    jt_sum = np.nansum(od_journey_time_sum) / 3600
    extr_jt_sum = np.nansum(od_extr_journey_time_mean) / 3600
    return od_journey_time_mean, od_journey_time_std, od_journey_time_rbt, od_count, jt_sum, extr_jt_sum


def get_departure_delay(trajectories_set, idx_dep_t, ordered_trip_ids, sched_departures):
    departure_delay = []
    for trajectories in trajectories_set:
        for trip_id in trajectories:
            trip_idx = ordered_trip_ids.index(trip_id)
            sched_dep = sched_departures[trip_idx]
            actual_dep = trajectories[trip_id][0][idx_dep_t]
            departure_delay.append(actual_dep - sched_dep)
    return departure_delay


def plot_travel_time_benchmark(tt_set, lbls, colors, pathname=None):
    i = 0
    for tt in tt_set:
        sns.kdeplot(np.array(tt), label=lbls[i], color=colors[i])
        i += 1
    plt.xlabel('total trip time (seconds)')
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_headway(hw_set, ordered_stops, lbls,
                 pathname=None, controlled_stops=None,
                 min_size_for_cv=1, cv_scale=(0, 1, 0.1)):
    fig, ax1 = plt.subplots()
    color = ['tab:red', 'tab:blue', 'tab:green']
    j = 0
    for hs in hw_set:
        x = []
        y1 = []
        for i in range(len(ordered_stops)):
            s = ordered_stops[i]
            if s in hs:
                h = hs[s]
                if len(h) > min_size_for_cv:
                    mean = np.array(h).mean()
                    std = np.array(h).std()
                    cv = std / mean
                    x.append(i)
                    y1.append(cv) if mean else y1.append(0)
        ax1.plot(x, y1, color=color[j], label=lbls[j])
        j += 1
    ax1.set_xlabel('stop id')
    ax1.set_ylabel('coefficient of variation of headway')
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]))
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x1 = np.arange(len(ordered_stops))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ordered_stops, fontsize=6, rotation=90)
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_headway_benchmark(hw_set, ordered_stops, lbls, colors, pathname=None, controlled_stops=None,
                           min_size_for_cv=1, cv_scale=(0, 1, 0.1)):
    fig, ax1 = plt.subplots()

    j = 0
    for hs in hw_set:
        x = []
        y1 = []
        for i in range(len(ordered_stops)):
            s = ordered_stops[i]
            if s in hs:
                h = hs[s]
                if len(h) > min_size_for_cv:
                    mean = np.array(h).mean()
                    std = np.array(h).std()
                    cv = std / mean
                    x.append(i)
                    y1.append(cv) if mean else y1.append(0)

        ax1.plot(x, y1, color=colors[j], label=lbls[j])
        j += 1

    ax1.set_xlabel('stop id')
    ax1.set_ylabel('coefficient of variation of headway')
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]))
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x1 = np.arange(len(ordered_stops))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ordered_stops, fontsize=6, rotation=90)
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_load_profile_benchmark(load_set, os, lbls, colors, pathname=None, x_y_lbls=None, controlled_stops=None):
    x1 = np.arange(len(os))
    fig, ax1 = plt.subplots()
    j = 0
    for load_profile in load_set:
        y = []
        for stop in os:
            if stop in load_profile:
                y.append(load_profile[stop])
            else:
                y.append(0)
        ax1.plot(x1, y, label=lbls[j], color=colors[j])
        j += 1
    if controlled_stops:
        for cs in controlled_stops:
            idx = os.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(os, fontsize=6, rotation=90)
    # right, left, top, bottom
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1])
    fig.legend(loc='upper center')
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_mean_hold_time_benchmark(ht_set, lbl, colors, pathname=None):
    w = 0.3
    for i in range(len(ht_set)):
        plt.bar(w*i, ht_set[i], label=lbl[i], color=colors[i])
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_hold_time_distribution_benchmark(ht_set, lbl, colors, pathname=None):
    i = 0
    for ht in ht_set:
        sns.kdeplot(np.array(ht), label=lbl[i], color=colors[i])
        i += 1
    plt.xlabel('seconds')
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_denied_benchmark(db_set, lbl, colors, pathname=None):
    w = 0.15
    for i in range(len(db_set)):
        plt.bar(w*i, db_set[i], label=lbl[i], color=colors[i])
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_wait_time_benchmark(wt_set, os, lbl, colors, pathname=None):
    fig, ax = plt.subplots()
    j = 0
    for wt in wt_set:
        ax.bar(np.arange(len(wt)), wt, label=lbl[j], color=colors[j])
        j += 1
    ax.set_xticks(np.arange(len(os)))
    ax.set_xticklabels(os, fontsize=6, rotation=90)
    ax.set_xlabel('stop')
    ax.set_ylabel('wait time (seconds)')
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return

