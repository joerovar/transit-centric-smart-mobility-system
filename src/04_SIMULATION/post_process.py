import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from copy import deepcopy
import seaborn as sns
from datetime import timedelta
import matplotlib


def write_trajectory_set(trajectory_set, pathname, idx_arr_t, idx_dep_t, idx_hold, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trajectories in trajectory_set:
            day = i
            for trip in trajectories:
                for stop_info in trajectories[trip]:
                    stop_lst = deepcopy(stop_info)
                    stop_lst[idx_arr_t] = str(timedelta(seconds=round(stop_lst[idx_arr_t])))
                    stop_lst[idx_dep_t] = str(timedelta(seconds=round(stop_lst[idx_dep_t])))
                    stop_lst[idx_hold] = round(stop_lst[idx_hold])
                    stop_lst.insert(0, trip)
                    stop_lst.append(day)
                    stop_lst.append(round(stop_info[idx_arr_t]))
                    stop_lst.append(round(stop_info[idx_dep_t]))
                    stop_lst.append(round(stop_info[idx_dep_t]-stop_info[idx_arr_t]))
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


def plot_load_profile(bd, al, lp, os, through, pathname=None, x_y_lbls=None, controlled_stops=None):
    w = 0.5
    x1 = np.arange(len(os))
    x2 = [i + w for i in x1]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(x1, bd, w, label='ons', color='darkgreen')
    ax1.bar(x2, al, w, label='offs', color='palegreen')
    ax2.plot(x1, lp, label='load', color='dodgerblue')
    ax2.plot(x1, through, label='through', color='darkturquoise', linestyle='dashed')
    if controlled_stops:
        for cs in controlled_stops:
            idx = os.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x_ticks = np.arange(0, len(os), 5)
    x_tick_labels = x_ticks + 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    # ax1.set_xticklabels(x_ticks, fontsize=6, rotation=90)
    # right, left, top, bottom
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1], color='darkgreen')
        ax1.tick_params(axis='y', colors='darkgreen')
        ax2.set_ylabel(x_y_lbls[2], color='dodgerblue')
        ax2.tick_params(axis='y', colors='dodgerblue')
    plt.tight_layout()
    plt.grid(axis='y')
    fig.legend(loc='upper center')
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def get_headway_from_trajectory_set(trajectory_set, idx_arr_t, stops, controlled_stops=None):
    cv_hw = {s: [] for s in stops}
    cv_hw_per_stop = []
    hw_at_tp = []
    cv_hw_mean = []
    i = 0
    for trajectories in trajectory_set:
        recorded_hw = {s: [] for s in stops}
        prev_stop_time = {}
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                stop_arr_time = stop_details[idx_arr_t]
                if stop_id not in prev_stop_time:
                    prev_stop_time[stop_id] = stop_arr_time
                else:
                    t1 = prev_stop_time[stop_id]
                    t2 = stop_arr_time
                    headway = t2 - t1
                    recorded_hw[stop_id].append(headway)
                    if stop_id in controlled_stops:
                        hw_at_tp.append(headway)
                    prev_stop_time[stop_id] = stop_arr_time
        for s in recorded_hw:
            headways = np.array(recorded_hw[s])
            # print(headways)
            cv_hw[s].append(headways.std() / headways.mean())
        cv_rep = []
        for s in cv_hw:
            cv_rep.append(cv_hw[s][-1])
        cv_hw_mean.append(sum(cv_rep) / len(cv_rep))
        i += 1
    for s in stops:
        cv_hw_per_stop.append(np.mean(cv_hw[s]))
    return cv_hw_per_stop, hw_at_tp, cv_hw_mean


def pax_per_trip_from_trajectory_set(trajectory_set, idx_load, idx_ons, idx_offs, stops):
    avg_lp = {s: [] for s in stops}
    avg_lp_sd = {s: [] for s in stops}
    avg_ons = {s: [] for s in stops}
    avg_offs = {s: [] for s in stops}
    avg_lp_per_stop = []
    avg_lp_sd_per_stop = []
    avg_ons_per_stop = []
    avg_offs_per_stop = []
    tot_ons_per_replication = []
    for trajectories in trajectory_set:
        bus_load_all = {s: [] for s in stops}
        ons_all = {s: [] for s in stops}
        offs_all = {s: [] for s in stops}
        total_ons = 0
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                bus_load = stop_details[idx_load]
                ons = stop_details[idx_ons]
                total_ons += ons
                offs = stop_details[idx_offs]
                bus_load_all[stop_id].append(bus_load)
                ons_all[stop_id].append(ons)
                offs_all[stop_id].append(offs)
        tot_ons_per_replication.append(total_ons)
        for stop in bus_load_all:
            avg_lp[stop].append(np.mean(bus_load_all[stop]))
            avg_lp_sd[stop].append(np.std(bus_load_all[stop]))
            avg_ons[stop].append(np.mean(ons_all[stop]))
            avg_offs[stop].append(np.mean(offs_all[stop]))
    avg_tot_ons = np.mean(tot_ons_per_replication)
    for s in stops:
        avg_lp_per_stop.append(np.mean(avg_lp[s]))
        avg_lp_sd_per_stop.append(np.mean(avg_lp_sd[s]))
        avg_ons_per_stop.append(np.mean(avg_ons[s]))
        avg_offs_per_stop.append(np.mean(avg_offs[s]))
    return avg_lp_per_stop, avg_lp_sd_per_stop, avg_tot_ons, avg_ons_per_stop, avg_offs_per_stop


def hold_time_from_trajectory_set(trajectory_set, idx, idx_stop, controlled_stops):
    tot_hold_times_mean = []
    all_ht_per_stop = [[] for cs in controlled_stops]
    for trajectories in trajectory_set:
        tot_hold_times = []
        for trip in trajectories:
            trip_hold_time = 0
            for stop_details in trajectories[trip]:
                ht = stop_details[idx]
                trip_hold_time += ht
                if stop_details[idx_stop] in controlled_stops:
                    k = controlled_stops.index(stop_details[idx_stop])
                    all_ht_per_stop[k].append(ht)
            tot_hold_times.append(trip_hold_time)
        tot_hold_times_mean.append(int(np.mean(tot_hold_times)))
    ht_per_stop = []
    for k in all_ht_per_stop:
        ht_per_stop.append(np.mean(k))
    tot_hold_time_mean = sum(tot_hold_times_mean) / len(tot_hold_times_mean)
    return tot_hold_time_mean, ht_per_stop


def denied_from_trajectory_set(trajectory_set, idx, avg_tot_ons):
    tot_denied_per_replication = []
    for trajectories in trajectory_set:
        tot_denied = 0
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                denied = stop_details[idx]
                tot_denied += denied
        tot_denied_per_replication.append(tot_denied)
    avg_tot_denied = np.mean(tot_denied_per_replication)
    per_mil_denied = avg_tot_denied * 1000 / avg_tot_ons
    return per_mil_denied


def tot_trip_times_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t):
    tot_trip_times = []
    extreme_trip_times = []
    rep_nr = 1
    for trajectories in trajectory_set:
        lst_trips = list(trajectories.keys())
        trip_times = [trajectories[k][-1][idx_arr_t] - trajectories[k][0][idx_dep_t] for k in lst_trips]
        extreme_trip_times.append(np.percentile(trip_times, 95))
        trip_times_long = [t for t in trip_times if t > 5500]
        if trip_times_long:
            print('----')
            print(rep_nr)
            print(trip_times_long)
        tot_trip_times += trip_times
        rep_nr += 1
    extreme_trip_times_mean = np.mean(extreme_trip_times)
    return tot_trip_times, extreme_trip_times_mean


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


def get_wait_times(pax_set, ordered_stops):
    stop_wait_time_mean = np.zeros(shape=(len(pax_set), len(ordered_stops)))
    stop_wait_time_mean[:] = np.nan
    i = 0
    for replication in pax_set:
        wait_times = {'o': [], 'd': [], 'wt': [], 'bt': [], 'at': []}
        for p in replication:
            wait_times['o'].append(p.orig_idx)
            wait_times['d'].append(p.dest_idx)
            wait_times['wt'].append(p.wait_time)
            wait_times['at'].append(str(timedelta(seconds=round(p.arr_time))))
            wait_times['bt'].append(str(timedelta(seconds=round(p.board_time))))
        wait_times_df = pd.DataFrame(wait_times)
        for j in range(len(ordered_stops)):
            wt = wait_times_df[(wait_times_df['o'] == j)]['wt']
            if not wt.empty:
                stop_wait_time_mean[i, j] = wt.mean()
                # stop_wait_time_std[i] = wt.std()
        i += 1
    # print(stop_wait_time_mean[:, 25])
    stop_wait_time_mean = np.nanmean(stop_wait_time_mean, axis=0)
    return stop_wait_time_mean


def get_pax_times_fast(pax_set, n_stops):
    fields = ['orig_idx', 'dest_idx', 'journey_time', 'wait_time', 'denied']
    journey_time_set = []
    wait_time_set = []
    denied_rate_per_rep = []
    denied_wait_time_set = []
    extreme_jt_set = []
    rbt_od_set = []
    for rep in pax_set:
        df = pd.DataFrame([{f: getattr(p, f) for f in fields} for p in rep])
        journey_time_set.append(df['journey_time'].mean())
        extreme_jt_set.append(df['journey_time'].quantile(0.95))
        wait_time_set.append(df['wait_time'].mean())
        tot_pax = df.shape[0]
        denied_df = df[df['denied'] == 1]
        denied_pax = denied_df.shape[0]
        denied_wt = denied_df['wait_time'].mean()
        denied_rate_per_rep.append(denied_pax / tot_pax)
        denied_wait_time_set.append(denied_wt)

        rbt_od = np.zeros(shape=(n_stops, n_stops))
        rbt_od[:] = np.nan
        pax_count = 0
        for s0 in range(n_stops-1):
            o_jt = df[df['orig_idx'] == s0]
            for s1 in range(s0+1, n_stops):
                od_jt = o_jt[o_jt['dest_idx'] == s1]
                if not od_jt.empty:
                    pax_od = od_jt.shape[0]
                    pax_count += pax_od
                    rbt_od[s0, s1] = (od_jt['journey_time'].quantile(0.95) - od_jt['journey_time'].median()) * pax_od
        rbt_od_set.append(np.nansum(rbt_od) / pax_count)
    # journey_time_mean = np.mean(journey_time_set)
    # wait_time_mean = np.mean(wait_time_set)
    denied_rate = np.mean(denied_rate_per_rep)
    denied_wait_time_mean = np.nanmean(denied_wait_time_set)
    # extreme_jt_mean = np.nanmean(extreme_jt_set)
    # jt_od_mean = np.mean(jt_od_set)
    return journey_time_set, wait_time_set, denied_rate, denied_wait_time_mean, rbt_od_set


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


def plot_headway_benchmark(cv_hw_set, ordered_stops, lbls, colors, pathname=None, controlled_stops=None,
                           cv_scale=(0, 1, 0.1)):
    fig, ax1 = plt.subplots()
    x = np.arange(len(ordered_stops))
    j = 0
    for cv in cv_hw_set:
        ax1.plot(x, cv, color=colors[j], label=lbls[j])
        j += 1

    ax1.set_xlabel('stop', fontsize=8)
    ax1.set_ylabel('coefficient of variation of headway', fontsize=8)
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]))
    ax1.set_yticklabels(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]).round(decimals=1), fontsize=8)
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(ordered_stops, fontsize=6, rotation=90)
    x_ticks = np.arange(0, len(ordered_stops), 5)
    x_tick_labels = x_ticks + 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, fontsize=8)
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
        # y = []
        # for stop in os:
        #     if stop in load_profile:
        #         y.append(load_profile[stop])
        #     else:
        #         y.append(0)
        ax1.plot(x1, load_profile, label=lbls[j], color=colors[j])
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
    x = [w*i for i in range(len(db_set))]
    for i in range(len(db_set)):
        plt.bar(x[i], db_set[i], label=lbl[i], color=colors[i], width=w)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    plt.ylabel('denied per thousand boardings')
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_wait_time_benchmark(wt_set, os, lbl, colors, pathname=None, scheduled_wait=None, controlled_stops=None):
    fig, ax = plt.subplots()
    j = 0
    for wt in wt_set:
        ax.plot(np.arange(len(wt)), wt, label=lbl[j], color=colors[j])
        j += 1
    if scheduled_wait:
        plt.axhline(scheduled_wait, linestyle='dashed', color='gray')
    ax.set_xticks(np.arange(len(os)))
    ax.set_xticklabels(os, fontsize=6, rotation=90)
    ax.set_xlabel('stop')
    ax.set_ylabel('wait time (seconds)')
    if controlled_stops:
        for cs in controlled_stops:
            idx = os.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


