import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle


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


def plot_stop_headway(hs, pathname):
    for stop in hs:
        for h in hs[stop]:
            plt.scatter(stop, h, color='r', s=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_trajectories(trip_data, pathname):
    for trip in trip_data:
        td = np.array(trip_data[trip])
        if np.size(td):
            times = td[:, 1].astype(float)
            plt.plot(times, np.arange(len(times)))
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_multiple_bar_charts(wta, wtc, pathname, lbls):
    w = 0.27
    bar1 = np.arange(len(wta.keys()))
    bar2 = [i + w for i in bar1]
    plt.bar(bar1, wta.values(), w, label=lbls[0], color='b')
    plt.bar(bar2, wtc.values(), w, label=lbls[1], color='r')
    plt.xticks(bar1, wta.keys(), rotation=90, fontsize=6)
    plt.tight_layout()
    plt.legend()
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
            linktime = stop_data[i+1][1] - stop_data[i][1]
            link = stop_data[i][0] + '-' + stop_data[i+1][0]
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
    # mean_wait_time = {}
    # for stop in headway_data:
    #     mean_wait_time[stop] = round((np.array(headway_data[stop]).mean()) / 2, 1)
    wait_times = pd.DataFrame(mean_wait_time.items(), columns=['stop', 'wait_time_sec'])
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    wait_times = pd.merge(wait_times, s, on='stop')
    wait_times.to_csv(pathname, index=False)
    return


def plot_bar_chart(var, pathname):
    plt.bar(var.keys(), var.values())
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
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

