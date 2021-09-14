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


def plot_stop_headway(hs, pathname, y_scale=None):
    fig, ax = plt.subplots()
    for stop in hs:
        for h in hs[stop]:
            ax.scatter(stop, h, color='r', s=20)
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


def plot_trajectories(trip_data, pathname, stops):
    for trip in trip_data:
        td = np.array(trip_data[trip])
        if np.size(td):
            times = td[:, 1].astype(float)
            starting_stop = td[0, 0]
            starting_stop_idx = stops.index(starting_stop)
            y_axis = np.arange(starting_stop_idx, starting_stop_idx + len(times))
            plt.plot(times, y_axis)
    plt.xlabel('seconds')
    plt.ylabel('stops')
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_multiple_bar_charts(wta, wtc, pathname, lbls, x_y_lbls=None):
    w = 0.27
    bar1 = np.arange(len(wta.keys()))
    bar2 = [i + w for i in bar1]
    plt.bar(bar1, wta.values(), w, label=lbls[0], color='b')
    plt.bar(bar2, wtc.values(), w, label=lbls[1], color='r')
    plt.xticks(bar1, wta.keys(), rotation=90, fontsize=6)
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


def plot_bar_chart(var, pathname, x_y_lbls=None):
    plt.bar(var.keys(), var.values())
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


# def chop_trajectories(trajectories, start_time, end_time):
#   for trip in trajectories:
#         trajectory = trajectories[trip]
#         start_idx = 0
#         end_idx = -1
#         found_start = False
#         found_end = False
#         for i in range(len(trajectory)):
#             if trajectory[i][1] >= start_time:
#                 if not found_start:
#                     found_start = True
#                     start_idx = i
#             if trajectory[i][1] > end_time and not found_end:
#                 found_end = True
#                 end_idx = i - 1
#         trajectories[trip] = trajectory[start_idx:end_idx]
#     return trajectories


def get_historical_headway(pathname, dates, all_stops, trips):
    whole_df = pd.read_csv(pathname)
    all_stops = [int(s) for s in all_stops]
    df_period = whole_df[whole_df['trip_id'].isin(trips)]
    headway = {}
    for d in dates:
        df_temp = df_period[df_period['event_time'].astype(str).str[:10] == d]
        for s in all_stops:
            df_temp1 = df_temp[df_temp['stop_id'] == s]
            for i, j in zip(trips, trips[1:]):
                t2 = df_temp1[df_temp1['trip_id'] == j]
                t1 = df_temp1[df_temp1['trip_id'] == i]
                if (not t1.empty) & (not t2.empty):
                    hw = float(t2['avl_sec'])-float(t1['avl_sec'])
                    if hw < 0:
                        hw = 0
                    if s in headway:
                        headway[str(s)].append(hw)
                    else:
                        headway[str(s)] = [hw]
    return headway


def plot_boardings(pathname, arrival_rates, dem_interval_len):
    aggregated_boardings = {}
    for s in arrival_rates:
        arr = arrival_rates[s]
        agg = sum([a*dem_interval_len for a in arr])
        aggregated_boardings[s] = agg
    plot_bar_chart(aggregated_boardings, pathname)
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


def plot_cv(pathname, link_times_mean, link_times_sd):
    for link in link_times_mean:
        cvs = []
        for i in range(len(link_times_mean[link])):
            mean = link_times_mean[link][i]
            sd = link_times_sd[link][i]
            if mean and sd:
                cv = sd / mean
                cvs.append(cv)
        plt.scatter([link for i in range(len(cvs))], cvs, color='g', alpha=0.3, s=20)
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
