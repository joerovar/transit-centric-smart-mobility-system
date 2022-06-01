import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from copy import deepcopy
import seaborn as sns
from datetime import timedelta
from Input_Processor import get_interval, remove_outliers


def plot_learning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0", markersize=8)
    ax.set_xlabel("Training steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def validate_trip_t_outbound(avl_df, sim_df, start_time, end_time, stops, path_trip_t, path_dwell_t, dates):
    trip_t_avl, dwell_t_avl = trip_t_outbound(avl_df, start_time, end_time, 60, stops, 'avl_arr_sec',
                                              'avl_dep_sec', is_avl=True, dates=dates)
    trip_t_sim, dwell_t_sim = trip_t_outbound(sim_df, start_time, end_time, 60, stops, 'arr_sec',
                                              'dep_sec')
    plot_calib_hist(trip_t_avl, trip_t_sim, 5, path_trip_t, 'total trip time (seconds)')
    plot_calib_hist(dwell_t_avl, dwell_t_sim, 5, path_dwell_t, 'dwell time (seconds)')
    return


def trip_t_outbound(df_out, start_time, end_time, interval_length, stops_out, col_arr_t, col_dep_t,
                    is_avl=False, dates=None):
    focus_df_out = df_out[df_out['stop_sequence'] == 1].copy()
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] < end_time]
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] >= start_time]
    if is_avl:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == int(stops_out[0])]
    else:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == stops_out[0]]
    focus_df_out = focus_df_out.sort_values(by='schd_sec')
    focus_trips = focus_df_out['trip_id'].unique().tolist()

    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    trip_t = [[] for _ in range(interval0, interval1)]
    dwell_t = [[] for _ in range(interval0, interval1)]
    if is_avl:
        nr_days = len(dates)
    else:
        nr_days = df_out['replication'].max()
    for i in range(nr_days):
        if is_avl:
            day_df = df_out[df_out['avl_arr_time'].astype(str).str[:10] == dates[i]]
        else:
            day_df = df_out[df_out['replication'] == i + 1]
        for trip in focus_trips:
            trip_df = day_df[day_df['trip_id'] == trip]
            t0 = trip_df[trip_df['stop_sequence'] == 1]
            t1 = trip_df[trip_df['stop_sequence'] == 67]
            if not t0.empty and not t1.empty:
                t0 = t0.iloc[0]
                t1 = t1.iloc[0]
                interval = get_interval(t0['schd_sec'], interval_length)
                dep_t = t0[col_dep_t].astype(int)
                arr_t = t1[col_arr_t].astype(int)
                trip_t[interval - interval0].append(arr_t - dep_t)

                mid_route_df = trip_df[(trip_df['stop_sequence'] != 1) & (trip_df['stop_sequence'] != 67)]
                if is_avl:
                    mid_route_df = mid_route_df.drop_duplicates(subset='stop_sequence', keep='first')
                if mid_route_df.shape[0] == 67 - 2:
                    mid_route_df['dwell_t'] = mid_route_df[col_dep_t] - mid_route_df[col_arr_t]
                    dwell_t[interval - interval0].append(mid_route_df['dwell_t'].sum())
    if is_avl:
        for i in range(interval1 - interval0):
            if trip_t[i]:
                trip_t[i] = remove_outliers(np.array(trip_t[i])).tolist()
            if dwell_t[i]:
                dwell_t[i] = remove_outliers(np.array(dwell_t[i])).tolist()
    return trip_t, dwell_t


def validate_delay_inbound(avl_df, sim_df, start_t_sec, end_t_sec, start_interval, interval_mins=60):
    start_interval = 5
    arr_delays_long, arr_delays_short, dep_delays_long, dep_delays_short = delay_inbound(avl_df, start_t_sec,
                                                                                         end_t_sec,
                                                                                         interval_mins,
                                                                                         'avl_arr_sec',
                                                                                         'avl_dep_sec',
                                                                                         [('15136', 1),
                                                                                          ('386', 23)],
                                                                                         [('8613', 1),
                                                                                          ('386', 63)],
                                                                                         outlier_removal=True)

    arr_del_long_sim, arr_del_short_sim, dep_del_long_sim, dep_del_short_sim = delay_inbound(sim_df,
                                                                                             start_t_sec,
                                                                                             end_t_sec,
                                                                                             interval_mins,
                                                                                             'arr_sec',
                                                                                             'arr_sec',
                                                                                             [('15136', 1),
                                                                                              ('386', 23)],
                                                                                             [('8613', 1),
                                                                                              ('386', 63)])

    plot_calib_hist(arr_delays_long, arr_del_long_sim, start_interval, 'out/compare/validate/arr_delays_in_long.png',
                    'arr delay (seconds)')
    plot_calib_hist(arr_delays_short, arr_del_short_sim, start_interval, 'out/compare/validate/arr_delays_in_short.png',
                    'arr delay (seconds)')
    plot_calib_hist(dep_delays_long, dep_del_long_sim, start_interval, 'out/compare/validate/dep_delays_in_long.png',
                    'dep delay (seconds)')
    plot_calib_hist(dep_delays_short, dep_del_short_sim, start_interval, 'out/compare/validate/dep_delays_in_short.png',
                    'dep delay (seconds)')
    return


def validate_delay_outbound(avl_df, sim_df, start_t_sec, end_t_sec, interval_mins=60):
    start_interval = 5
    arr_delays_out, dep_delays_out = delay_outbound(avl_df, start_t_sec, end_t_sec, interval_mins,
                                                    'avl_arr_sec', 'avl_dep_sec', [('386', 1), ('8613', 67)],
                                                    outlier_removal=True)
    arr_delays_out_sim, dep_delays_out_sim = delay_outbound(sim_df, start_t_sec, end_t_sec,
                                                            interval_mins, 'arr_sec', 'dep_sec',
                                                            [('386', 1), ('8613', 67)])
    plot_calib_hist(arr_delays_out, arr_delays_out_sim, start_interval, 'out/compare/validate/arr_delays_out.png',
                    'arr delay (seconds)')
    plot_calib_hist(dep_delays_out, dep_delays_out_sim, start_interval, 'out/compare/validate/dep_delays_out.png',
                    'dep delay (seconds)')
    return


def plot_calib_hist(delay_avl, delay_sim, start_interval, filename, xlabel):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in range(ax.size):
        ax.flat[i].hist([delay_avl[i], delay_sim[i]], density=True, label=['avl', 'sim'])
        ax.flat[i].set_title(f'hour {start_interval + i}')
        ax.flat[i].set_yticks([])
        ax.flat[i].set_xlabel(xlabel)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()
    return


def validate_cv_hw_outbound(avl_df, sim_df, start_t_sec, end_t_sec, interval_min, stops, dates, start_interval=5):
    hw_out_cv = cv_hw_from_avl(avl_df, start_t_sec, end_t_sec, interval_min, stops, dates)
    hw_out_cv_sim = cv_hw(sim_df, start_t_sec, end_t_sec, interval_min, stops)
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='all', sharex='all')
    for i in range(ax.size):
        ax.flat[i].plot(hw_out_cv[i], label='avl')
        ax.flat[i].plot(hw_out_cv_sim[i], label='sim')
        ax.flat[i].set_title(f'hour {start_interval+i}')
        ax.flat[i].set_xlabel('stop')
    ax[0,0].set_ylabel('c.v. headway')
    ax[1,0].set_ylabel('c.v. headway')
    plt.legend()
    plt.tight_layout()
    plt.savefig('out/compare/validate/cv_hw.png')
    plt.close()
    return


def delay_inbound(trips_df, start_time, end_time, delay_interval_length, col_arr_t, col_dep_t, terminals_short,
                  terminals_long, outlier_removal=False):
    trips_df2 = trips_df.copy()
    trips_df2['stop_id'] = trips_df2['stop_id'].astype(str)

    end_terminal_id = terminals_short[1][0]
    terminal_seq_long = terminals_long[1][1]
    terminal_seq_short = terminals_short[1][1]

    arr_delays_long = []
    arr_delays_short = []
    # arrivals
    arr_long_df = trips_df2[trips_df2['stop_id'] == end_terminal_id]
    arr_long_df = arr_long_df[arr_long_df['stop_sequence'] == terminal_seq_long]
    arr_long_df[col_arr_t] = arr_long_df[col_arr_t] % 86400

    arr_short_df = trips_df2[trips_df2['stop_id'] == end_terminal_id]
    arr_short_df = arr_short_df[arr_short_df['stop_sequence'] == terminal_seq_short]
    arr_short_df[col_arr_t] = arr_short_df[col_arr_t] % 86400

    start_terminal_id_long = terminals_long[0][0]
    start_terminal_id_short = terminals_short[0][0]

    dep_delays_long = []
    dep_delays_short = []
    # departures
    dep_long_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_long]
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == 1]
    dep_long_df[col_dep_t] = dep_long_df[col_dep_t] % 86400

    dep_short_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_short]
    dep_short_df = dep_short_df[dep_short_df['stop_sequence'] == 1]
    dep_short_df[col_dep_t] = dep_short_df[col_dep_t] % 86400

    interval0 = get_interval(start_time, delay_interval_length)
    interval1 = get_interval(end_time, delay_interval_length)

    for interval in range(interval0, interval1):
        # arrivals
        temp_df = arr_long_df[arr_long_df['schd_sec'] >= interval*delay_interval_length*60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval+1)*delay_interval_length*60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays_long.append(d.tolist())

        temp_df = arr_short_df[arr_short_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays_short.append(d.tolist())

        # departures
        temp_df = dep_long_df[dep_long_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays_long.append(d.tolist())

        temp_df = dep_short_df[dep_short_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays_short.append(d.tolist())
    return arr_delays_long, arr_delays_short, dep_delays_long, dep_delays_short


def delay_outbound(trips_df, start_time, end_time, delay_interval_length, col_arr_t, col_dep_t, terminals_info,
                   outlier_removal=False):
    trips_df2 = trips_df.copy()
    trips_df2['stop_id'] = trips_df2['stop_id'].astype(str)

    end_terminal_id = terminals_info[1][0]
    terminal_seq_long = terminals_info[1][1]

    arr_delays = []
    # arrivals
    arr_long_df = trips_df2[trips_df2['stop_id'] == end_terminal_id]
    arr_long_df = arr_long_df[arr_long_df['stop_sequence'] == terminal_seq_long]
    arr_long_df[col_arr_t] = arr_long_df[col_arr_t] % 86400
    start_terminal_id_long = terminals_info[0][0]

    dep_delays = []
    # departures
    dep_long_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_long]
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == 1]
    dep_long_df[col_dep_t] = dep_long_df[col_dep_t] % 86400

    interval0 = get_interval(start_time, delay_interval_length)
    interval1 = get_interval(end_time, delay_interval_length)

    for interval in range(interval0, interval1):
        # arrivals
        temp_df = arr_long_df[arr_long_df['schd_sec']>=interval*delay_interval_length*60]
        temp_df = temp_df[temp_df['schd_sec']<=(interval+1)*delay_interval_length*60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays.append(d.tolist())

        # departures
        temp_df = dep_long_df[dep_long_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays.append(d.tolist())
    return arr_delays, dep_delays


def cv_hw(trip_record_df, start_time, end_time, interval_length, stops):
    nr_replications = trip_record_df['replication'].max()
    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    hws = [[[] for _ in range(len(stops))] for _ in range(interval0, interval1)]
    for rep_nr in range(1, nr_replications+1):
        date_df = trip_record_df[trip_record_df['replication'] == rep_nr]
        for interval in range(interval0, interval1):
            temp_df = date_df[date_df['schd_sec'] >= interval * interval_length * 60]
            temp_df = temp_df[temp_df['schd_sec'] < (interval + 1) * interval_length * 60]
            for j in range(len(stops)):
                df = temp_df[temp_df['stop_id'] == stops[j]]
                df = df[df['stop_sequence'] == j+1]
                df = df.sort_values(by='arr_sec')
                arr_sec = df['arr_sec'].tolist()
                if len(arr_sec) > 1:
                    for i in range(1, len(arr_sec)):
                        hws[interval-interval0][j].append(arr_sec[i] - arr_sec[i-1])
    cv_hws = []
    for interval in range(len(hws)):
        cv_hws.append([])
        for stop_idx in range(len(hws[interval])):
            cv_hws[-1].append(np.std(hws[interval][stop_idx]) / np.mean(hws[interval][stop_idx]))
    return cv_hws


def cv_hw_from_avl(avl_df, start_time, end_time, interval_length, stops,  dates):
    avl_df2 = avl_df.copy()
    avl_df2['stop_id'] = avl_df2['stop_id'].astype(str)
    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    hws = [[[] for _ in range(len(stops))] for _ in range(interval0, interval1)]
    for d in dates:
        date_df = avl_df2[avl_df2['avl_arr_time'].astype(str).str[:10] == d]
        for interval in range(interval0, interval1):
            temp_df = date_df[date_df['schd_sec'] >= interval * interval_length * 60]
            temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * interval_length * 60]
            for j in range(len(stops)):
                df = temp_df[temp_df['stop_id'] == stops[j]]
                df = df[df['stop_sequence'] == j+1]
                df = df.sort_values(by='avl_arr_sec')
                arr_sec = df['avl_arr_sec'].tolist()
                if len(arr_sec) > 1:
                    for i in range(1, len(arr_sec)):
                        hws[interval - interval0][j].append(arr_sec[i] - arr_sec[i-1])
    cv_hws = []
    for interval in range(len(hws)):
        cv_hws.append([])
        for stop_idx in range(len(hws[interval])):
            hws_arr = remove_outliers(np.array(hws[interval][stop_idx]))
            cv_hws[-1].append(np.std(hws_arr) / np.mean(hws_arr))
    return cv_hws


def save(pathname, par):
    with open(pathname, 'wb') as tf:
        pickle.dump(par, tf)
    return


def load(pathname):
    with open(pathname, 'rb') as tf:
        var = pickle.load(tf)
    return var


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


def write_trajectory_set_inbound(trajectory_set, pathname, idx_arr_t, idx_schd_sec, header=None):
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
                    stop_lst[idx_schd_sec] = str(timedelta(seconds=round(stop_lst[idx_schd_sec])))
                    stop_lst.insert(0, trip)
                    stop_lst.append(day)
                    stop_lst.append(round(stop_info[idx_arr_t]))
                    stop_lst.append(round(stop_info[idx_schd_sec]))
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
    fig, axs = plt.subplots(ncols=3, sharey='all')
    for i in range(3):
        for trip in trip_data[i]:
            td = np.array(trip_data[i][trip])[:52]
            if np.size(td):
                arr_times = td[:, idx_arr_t].astype(float)
                dep_times = td[:, idx_dep_t].astype(float)
                times = np.vstack((arr_times, dep_times))
                times = times.flatten(order='F')
                # print(times)
                starting_stop = td[0, 0]
                starting_stop_idx = ordered_stops.index(starting_stop)
                y_axis = np.arange(starting_stop_idx, starting_stop_idx + len(arr_times))
                y_axis = np.repeat(y_axis, 2)
                axs[i].plot(times, y_axis, color='lightblue')
        if controlled_stops:
            for c in controlled_stops[:-1]:
                stop_idx = ordered_stops.index(c)
                axs[i].axhline(y=stop_idx, color='gray', alpha=0.5, linestyle='dashed')
    # axs[0].set_yticks(np.arange(len(ordered_stops)))
    axs[0].tick_params(axis='both', labelsize=6)
    axs[1].tick_params(axis='x', labelsize=6)
    axs[2].tick_params(axis='x', labelsize=6)
    axs[0].set(xlabel='seconds', ylabel='stops')
    # plt.tick_params(labelright=True)
    fig.tight_layout(pad=0.05)
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_pax_profile(bd, al, lp, os, through, pathname=None, x_y_lbls=None, controlled_stops=None):
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
            if idx == 0:
                plt.axvline(x=idx, color='black', alpha=0.7, linestyle='dashed', label='control stops')
            else:
                plt.axvline(x=idx, color='black', alpha=0.7, linestyle='dashed')
    x_ticks = np.arange(0, len(os), 5)
    x_tick_labels = x_ticks + 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    # right, left, top, bottom
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1], color='black')
        ax2.set_ylabel(x_y_lbls[2], color='black')
    plt.tight_layout()
    fig.legend(loc='upper center')
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def load_from_trajectory_set(trajectory_set, stops, idx_load, peak_load_stop):
    load_per_stop = {s: [] for s in stops}
    peak_loads = []
    for trajectories in trajectory_set:
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                bus_load = stop_details[idx_load]
                load_per_stop[stop_id].append(bus_load)
                if peak_load_stop == stop_id:
                    peak_loads.append(bus_load)
    load_avg_per_stop = [np.mean(load_per_stop[s]) for s in stops]
    load_sd_per_stop = [np.std(load_per_stop[s]) for s in stops]
    max_load_per_stop = [np.percentile(load_per_stop[s], 95) for s in stops]
    min_load_per_stop = [np.percentile(load_per_stop[s], 10) for s in stops]
    return load_avg_per_stop, load_sd_per_stop, peak_loads, max_load_per_stop, min_load_per_stop


def trip_time_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t):
    trip_times = []
    for trajectories in trajectory_set:
        for trip in trajectories:
            dep_time = trajectories[trip][1][idx_dep_t]
            arr_time = trajectories[trip][-2][idx_arr_t]
            trip_times.append(arr_time - dep_time)
    return trip_times


def plot_4_trip_t_dist(all_trip_t, tags, path_save):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    i = 0
    for temp_trip_t in all_trip_t:
        sns.histplot([t/60 for t in temp_trip_t], kde=True, color='gray', alpha=0.5, ax=axs.flat[i])
        axs.flat[i].axvline(np.percentile(temp_trip_t, 95)/60, color='black', linestyle='dashed', alpha=0.7)
        axs.flat[i].set_title(tags[i], fontsize=9)
        if i > 1:
            axs.flat[i].set_xlabel('total trip run time (min)', fontsize=8)
        i += 1
    plt.xlim(60, 81)
    for ax in axs.flat:
        ax.tick_params(labelsize=8)
        ax.set_ylabel('frequency', fontsize=8)
    # plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path_save + 'trip_t_dist.png')
    plt.close()
    return


def plot_5_trip_t_dist(all_trip_t, tags, path_save):
    fig, axs = plt.subplots(nrows=5, sharex='all', sharey='all')
    i = 0
    for temp_trip_t in all_trip_t:
        sns.histplot([t/60 for t in temp_trip_t], kde=True, color='gray', alpha=0.5, ax=axs.flat[i])
        axs.flat[i].axvline(np.percentile(temp_trip_t, 95)/60, color='black', linestyle='dashed', alpha=0.7)
        axs.flat[i].set_title(tags[i], fontsize=9)
        if i > 1:
            axs.flat[i].set_xlabel('total trip time (seconds)', fontsize=8)
        i += 1
    plt.xlim(60, 81)
    for ax in axs.flat:
        ax.tick_params(labelsize=8)
        ax.set_ylabel('frequency', fontsize=8)
    # plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path_save + 'trip_t_dist.png')
    plt.close()
    return


def control_from_trajectory_set(df_path, controlled_stops):
    df = pd.read_csv(df_path)
    df['stop_id'] = df['stop_id'].astype(str)
    df_control = df[df['stop_id'].isin(controlled_stops[:-1])]
    # print(df_control)
    hold_times = df_control['hold_time'].tolist()
    # nr_trips = len(df['trip_id'].unique().tolist()) * df['replication'].max()
    skipped_count = df_control['skipped'].sum()
    skipped_freq = skipped_count / df_control.shape[0] * 100
    return hold_times, skipped_freq


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


def get_stop_loc(pathname):
    stop_gps = pd.read_csv(pathname)
    stop_gps = stop_gps[['stop_id', 'stop_lat', 'stop_lon']]
    return stop_gps


def get_pax_times_fast(pax_set, n_stops, include_rbt=False):
    fields = ['orig_idx', 'dest_idx', 'journey_time', 'wait_time', 'denied']
    wait_time_set = []
    denied_rate_per_rep = []
    denied_wait_time_set = []
    rbt_od_set = []
    pax_wt_0_2 = 0
    pax_wt_2_4 = 0
    pax_wt_4_inf = 0
    for rep in pax_set:
        df = pd.DataFrame([{f: getattr(p, f) for f in fields} for p in rep])
        wait_time_set.append(df['wait_time'].mean() / 60)
        pax_wt_0_2 += df[df['wait_time'] < 2.5*60].shape[0]
        pax_wt_2_4 += df[(df['wait_time'] >= 2.5*60) & (df['wait_time'] < 5*60)].shape[0]
        pax_wt_4_inf += df[df['wait_time'] > 5*60].shape[0]
        tot_pax = df.shape[0]
        denied_df = df[df['denied'] == 1]
        denied_pax = denied_df.shape[0]
        denied_wt = denied_df['wait_time'].mean()
        denied_rate_per_rep.append(denied_pax / tot_pax)
        denied_wait_time_set.append(denied_wt)

        if include_rbt:
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
            rbt_od_set.append(np.nansum(rbt_od) / pax_count/60)
    denied_rate = np.mean(denied_rate_per_rep)
    denied_wait_time_mean = np.nanmean(denied_wait_time_set)
    tot_pax_reps = pax_wt_0_2 + pax_wt_2_4 + pax_wt_4_inf
    pc_pax_wt_0_2 = round(pax_wt_0_2 / tot_pax_reps * 100, 2)
    pc_pax_wt_2_4 = round(pax_wt_2_4 / tot_pax_reps * 100, 2)
    pc_pax_wt_4_inf = round(pax_wt_4_inf / tot_pax_reps * 100, 2)
    return wait_time_set, denied_rate, denied_wait_time_mean, rbt_od_set, pc_pax_wt_0_2, pc_pax_wt_2_4, pc_pax_wt_4_inf


def plot_headway(cv_hw_set, ordered_stops, lbls, colors, pathname=None, controlled_stops=None, cv_scale=(0, 1, 0.1)):
    fig, ax1 = plt.subplots()
    x = np.arange(len(ordered_stops))
    j = 0
    for cv in cv_hw_set:
        ax1.plot(x, cv, color=colors[j], label=lbls[j],marker='*')
        j += 1

    ax1.set_xlabel('stop', fontsize=8)
    ax1.set_ylabel('coefficient of variation of headway', fontsize=8)
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]))
    ax1.set_yticklabels(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]).round(decimals=1), fontsize=8)
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x_ticks = np.arange(0, len(ordered_stops), 3)
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


def plot_load_profile_grid(lp_set, lp_max_set, lp_min_set, os, tags, pathname=None):
    x1 = np.arange(len(os))
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    obj = []
    i = 0
    for ax in axs.flat:
        obj1, = ax.plot(x1, lp_set[i], color='black')
        obj2, = ax.plot(x1, lp_max_set[i], color='red')
        obj3, = ax.plot(x1, lp_min_set[i], color='green')
        obj.append([obj1,obj2,obj3])
        ax.set_title(tags[i], fontsize=9)
        ax.grid(axis='y')
        ax.axhline(y=50, color='red', alpha=0.5)
        ax.set_ylabel('load (pax)', fontsize=9)
        ax.set_xlabel('stop', fontsize=9)
        ax.tick_params(labelsize=9)
        i += 1
    fig.legend(obj[-1], ['median', '95-th', '10-th'], bbox_to_anchor=(0.535,0.0),loc='lower center', fontsize=9, ncol=3,
               columnspacing=0.8)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_load_profile_benchmark(load_set, os, lbls, colors, load_sd_set=None, pathname=None, x_y_lbls=None, controlled_stops=None):
    x1 = np.arange(len(os))
    fig, ax1 = plt.subplots()
    for j in range(len(load_set)):
        ax1.plot(x1, load_set[j], label=lbls[j], color=colors[j])
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


def plot_sensitivity_whisker(dset1, dset2, method_labels, scenario_labels, y_label1, y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    nr_methods = len(method_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='all')
    fig.subplots_adjust(wspace=0.02)
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[nr_methods*i:nr_methods*(i+1)], sym='')
        axes[0, i].set_xticks(np.arange(1, nr_methods+1))
        axes[0, i].set_xticklabels(method_labels, fontsize=8)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set(ylabel=y_label1)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[nr_methods*i:nr_methods*(i+1)], sym='')
        axes[1, i].set_xticks(np.arange(1, nr_methods+1))
        axes[1, i].set_xticklabels(method_labels, fontsize=8)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set(ylabel=y_label2)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_sensitivity_whisker_run_t(dset1, dset2, method_labels, scenario_labels, base_method_labels,
                                   y_label1, y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    nr_methods = len(method_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='col')
    fig.subplots_adjust(wspace=0.02)
    ranges = [(0, 5), (5, 8), (8, 13)]
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[ranges[i][0]:ranges[i][1]], sym='')
        axes[0, i].set_xticks(np.arange(1, ranges[i][1]-ranges[i][0]+1))
        # axes[0, i].set_xticklabels(method_labels, fontsize=8)
        if i == 1:
            axes[0, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[0, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set_ylabel(y_label1, fontsize=8)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[ranges[i][0]:ranges[i][1]], sym='')
        axes[1, i].set_xticks(np.arange(1, ranges[i][1]-ranges[i][0]+1))
        if i == 1:
            axes[1, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[1, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        # axes[1, i].set_xticklabels(method_labels, fontsize=8)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set_ylabel(y_label2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_sensitivity_whisker_compliance(dset1, dset2, method_labels, scenario_labels, base_method_labels,y_label1, y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='col')
    fig.subplots_adjust(wspace=0.02)
    ranges = [(0, 3), (3, 8), (8, 13)]
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[ranges[i][0]:ranges[i][1]], sym='')
        axes[0, i].set_xticks(np.arange(1, ranges[i][1]-ranges[i][0]+1))
        if i == 0:
            axes[0, i].set_xticklabels(base_method_labels, fontsize=8)
        else:
            axes[0, i].set_xticklabels(method_labels, fontsize=8)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set_ylabel(y_label1, fontsize=8)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[ranges[i][0]:ranges[i][1]], sym='')
        axes[1, i].set_xticks(np.arange(1, ranges[i][1]-ranges[i][0]+1))
        if i == 0:
            axes[1, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[1, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set_ylabel(y_label2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_2_var_whisker(var1, var2, tags, path_save, var1_label, var2_label, x_label=None):
    fig, axs = plt.subplots(ncols=2)
    axs[0].boxplot(var1, labels=tags, sym='', widths=0.2)
    axs[0].set_xticks(np.arange(1, len(tags)+1))
    axs[0].set_xticklabels(tags, fontsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(var1_label, fontsize=8)

    axs[1].boxplot(var2, labels=tags, sym='', widths=0.2)
    axs[1].set_xticks(np.arange(1, len(tags)+1))
    axs[1].set_xticklabels(tags, fontsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(var2_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_3_var_whisker(var1, var2, var3, tags, path_save, var1_label, var2_label, var3_label, x_label=None):
    fig, axs = plt.subplots(ncols=3)
    axs[0].boxplot(var1, labels=tags, sym='', widths=0.2)
    axs[0].set_xticks(np.arange(1, len(tags)+1))
    axs[0].set_xticklabels(tags, fontsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(var1_label, fontsize=8)

    axs[1].boxplot(var2, labels=tags, sym='', widths=0.2)
    axs[1].set_xticks(np.arange(1, len(tags)+1))
    axs[1].set_xticklabels(tags, fontsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(var2_label, fontsize=8)

    var3 = [[t/60 for t in v] for v in var3]
    axs[2].boxplot(var3, labels=tags, sym='', widths=0.2)
    axs[2].set_xticks(np.arange(1, len(tags)+1))
    axs[2].set_xticklabels(tags, fontsize=8)
    axs[2].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[2].set_xlabel(x_label)
    axs[2].set_ylabel(var3_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return

