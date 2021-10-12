import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
import math
import scipy.stats as stats


def write_trajectories(trip_data, pathname):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        for trip in trip_data:
            wf.writerow([trip, '{'])
            for stop in trip_data[trip]:
                wf.writerow(stop)
            wf.writerow([' ', '}'])
            wf.writerow('------')
            i += 1
    return


def plot_stop_headway(pathname, hs, ordered_stops, y_scale=None):
    fig, ax = plt.subplots()
    for stop in ordered_stops:
        if stop in hs:
            h = hs[stop]
            ax.scatter([stop] * len(h), h, color='r', s=20)
        else:
            ax.scatter(stop, np.nan)
    plt.xlabel('stop id')
    plt.ylabel('seconds')
    if y_scale:
        ax.set_ylim(y_scale)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_trajectories(trip_data, pathname, ordered_stops):
    for trip in trip_data:
        td = np.array(trip_data[trip])
        if np.size(td):
            times = td[:, 1].astype(float)
            starting_stop = td[0, 0]
            starting_stop_idx = ordered_stops.index(starting_stop)
            y_axis = np.arange(starting_stop_idx, starting_stop_idx + len(times))
            plt.plot(times, y_axis)
    plt.xlabel('seconds')
    plt.ylabel('stop sequence')
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


def plot_bar_chart(var, ordered_stops, pathname, x_y_lbls=None):
    x = ordered_stops
    y1 = []
    for s in ordered_stops:
        y1.append(var[s]) if s in var else y1.append(0)
    plt.bar(x, y1)
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


def write_link_times(trip_data, stop_gps, path_writename):
    link_times = {}
    for t in trip_data:
        stop_data = trip_data[t]
        for i in range(len(stop_data) - 1):
            linktime = stop_data[i + 1][1] - stop_data[i][1]
            link = stop_data[i][0] + '-' + stop_data[i + 1][0]
            if link in link_times:
                link_times[link].append(linktime)
            else:
                link_times[link] = [linktime]
    mean_link_times = {}
    for link in link_times:
        mean_link_times[link] = round(np.array(link_times[link]).mean(), 1)
    link_times_df = pd.DataFrame(mean_link_times.items(), columns=['stop_1', 'time_sec'])
    link_times_df[['stop_1', 'stop_2']] = link_times_df['stop_1'].str.split('-', expand=True)
    link_times_df = link_times_df[['stop_1', 'stop_2', 'time_sec']]
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop_1', 'stop_lat': 'stop_1_lat', 'stop_lon': 'stop_1_lon'})
    link_times_df = pd.merge(link_times_df, s, on='stop_1')
    s = s.rename(columns={'stop_1': 'stop_2', 'stop_1_lat': 'stop_2_lat', 'stop_1_lon': 'stop_2_lon'})
    link_times_df = pd.merge(link_times_df, s, on='stop_2')
    link_times_df.to_csv(path_writename, index=False)
    return


def write_wait_times(mean_wait_time, stop_gps, pathname):
    wait_times = pd.DataFrame(mean_wait_time.items(), columns=['stop', 'wait_time_sec'])
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    wait_times = pd.merge(wait_times, s, on='stop')
    wait_times.to_csv(pathname, index=False)
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


def merge_dictionaries(d1, d2, d3, d4):
    for k in d1:
        d1[k].extend(d2[k])
        d1[k].extend(d3[k])
        d1[k].extend(d4[k])
    return d1


def get_historical_headway(pathname, dates, all_stops, trips):
    whole_df = pd.read_csv(pathname)
    df_period = whole_df[whole_df['trip_id'].isin(trips)]
    headway = {}
    for d in dates:
        df_temp = df_period[df_period['event_time'].astype(str).str[:10] == d]
        for s in all_stops:
            df_temp1 = df_temp[df_temp['stop_id'] == int(s)]
            for i, j in zip(trips, trips[1:]):
                t2 = df_temp1[df_temp1['trip_id'] == j]
                t1 = df_temp1[df_temp1['trip_id'] == i]
                if (not t1.empty) & (not t2.empty):
                    hw = float(t2['avl_sec'])-float(t1['avl_sec'])
                    if hw < 0:
                        hw = 0
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


def write_travel_times(pathname, link_times_mean, link_times_std, nr_time_dpoints):
    with open(pathname, 'w') as f:
        fw = csv.writer(f)
        for key in link_times_mean:
            fw.writerow([key])
            fw.writerow(nr_time_dpoints[key])
            fw.writerow(link_times_mean[key])
            fw.writerow(link_times_std[key])
    return


def plot_cv(pathname, link_times_mean, link_times_sd):
    for link in link_times_mean:
        cvs = []
        for i in range(len(link_times_mean[link])):
            mean = link_times_mean[link][i]
            sd = link_times_sd[link][i]
            if mean and sd:
                cv = sd / mean
                cvs.append(cv)
        plt.scatter([link]*len(cvs), cvs, color='g', alpha=0.3, s=20)
    plt.ylabel('seconds')
    plt.xlabel('stop id')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def get_headway_from_trajectories(trajectories, idx_ons, idx_denied):
    prev_stop_time = {}
    prev_denied = {}
    recorded_headway = {}
    tot_wait_time = {}
    wait_time = {}
    wait_time_from_hw = {}
    tot_boardings = {}
    for trip in trajectories:
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
                    tot_wait_time[stop_id] = (prev_denied_ + (ons+denied-prev_denied_)*0.5) * headway
                    tot_boardings[stop_id] = ons
                else:
                    recorded_headway[stop_id].append(headway)
                    tot_wait_time[stop_id] += (prev_denied_ + (ons+denied-prev_denied_)*0.5) * headway
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


def count_from_trajectories(trajectories, idx, average=False):
    n = {}
    cntr = {}
    for trip in trajectories:
        for stop_details in trajectories[trip]:
            stop_id = stop_details[0]
            item_to_count = stop_details[idx]
            if stop_id not in n:
                n[stop_id] = item_to_count
                if average:
                    cntr[stop_id] = 1
            else:
                n[stop_id] += item_to_count
                if average:
                    cntr[stop_id] += 1
    if average:
        for stop in n:
            n[stop] = n[stop] / cntr[stop] if cntr[stop] else 0
    return n


def plot_pax_per_stop(pathname, pax, ordered_stops, x_y_lbls):
    for stop in ordered_stops:
        if stop in pax:
            plt.bar(stop, pax[stop], color='b')
        else:
            plt.bar(stop, np.nan, color='b')
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


def plot_load_profile(bd, al, l, os, pathname=None, x_y_lbls=None):
    w = 0.27
    x1 = np.arange(len(os))
    x2 = [i + w for i in x1]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    y1, y2, y3 = [], [], []
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
    ax1.bar(x1, y1, w, label='ons', color='peru')
    ax1.bar(x2, y2, w, label='offs', color='peachpuff')
    ax2.plot(x1, y3, label='load', color='dodgerblue')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(os, fontsize=6, rotation=90)
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1], color='peru')
        ax2.set_ylabel(x_y_lbls[2], color='dodgerblue')
    plt.tight_layout()
    fig.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return

