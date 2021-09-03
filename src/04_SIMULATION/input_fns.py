import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def get_interval(t, len_i):
    interval = int(t/(len_i*60))
    return interval


def get_route(path_stop_times, start_time, end_time, nr_intervals, start_interval, interval_length, dates, trip_choice):
    link_times = {}
    route_stop_times = pd.read_csv(path_stop_times)
    whole_df = pd.read_csv(path_stop_times)
    df = whole_df[whole_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['avl_sec'] % 86400 <= end_time]
    df = df[df['avl_sec'] % 86400 >= start_time]
    trip_ids = df['trip_id'].unique().tolist()
    for t in trip_ids:
        temp = route_stop_times[route_stop_times['trip_id'] == t]
        temp = temp.sort_values(by='stop_sequence')
        for d in dates:
            date_specific = temp[temp['event_time'].astype(str).str[:10] == d]
            stop_id = date_specific['stop_id'].astype(str).tolist()
            times = date_specific['avl_sec'].tolist()
            check_sequence = date_specific['stop_sequence'].tolist()
            if times:
                for i in range(len(stop_id)-1):
                    if check_sequence[i] == check_sequence[i + 1] - 1:
                        link = stop_id[i]+'-'+stop_id[i+1]
                        exists = link in link_times
                        if not exists:
                            link_times[link] = [[] for i in range(nr_intervals)]
                        nr_bin = get_interval(times[i] % 86400, interval_length) - start_interval
                        if 0 <= nr_bin < nr_intervals:
                            link_times[link][nr_bin].append(times[i+1] - times[i])
    mean_link_times = {}
    stdev_link_times = {}
    nr_dpoints_link_times = {}
    for link in link_times:
        mean_link_times[link] = []
        stdev_link_times[link] = []
        nr_dpoints_link_times[link] = []
        for b in link_times[link]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_link_times[link].append(np.array(b).mean())
                stdev_link_times[link].append(np.array(b).std())
                nr_dpoints_link_times[link].append(len(b))
    df_forstops = route_stop_times[route_stop_times['trip_id'] == trip_choice]
    df_forstops = df_forstops[df_forstops['event_time'].astype(str).str[:10] == dates[0]]
    df_forstops = df_forstops.sort_values(by='stop_sequence')
    all_stops = df_forstops['stop_id'].astype(str).tolist()
    return all_stops, mean_link_times, stdev_link_times, nr_dpoints_link_times


def get_demand(path, stops, nr_intervals, interval_length, start_interval):
    arr_rates = {}
    drop_rates = {}
    alight_fractions = {}
    # first we turn it into an OD matrix
    od_pairs = pd.read_csv(path)
    viable_dest = {}
    viable_orig = {}
    # since stops are ordered, stop n is allowed to pair with stop n+1 until N
    for i in range(len(stops)):
        if i == 0:
            viable_dest[stops[i]] = stops[i+1:]
        elif i == len(stops) - 1:
            viable_orig[stops[i]] = stops[:i]
        else:
            viable_dest[stops[i]] = stops[i + 1:]
            viable_orig[stops[i]] = stops[:i]
    for s in stops:
        # record arrival rate
        arrivals = od_pairs[od_pairs['BOARDING_STOP'].astype(str).str[:-2] == s].reset_index()
        arrivals = arrivals[arrivals['INFERRED_ALIGHTING_GTFS_STOP'].astype(str).str[:-2].isin(viable_dest.get(s, []))].reset_index()
        # record drop-offs
        dropoffs = od_pairs[od_pairs['INFERRED_ALIGHTING_GTFS_STOP'].astype(str).str[:-2] == s].reset_index()
        dropoffs = dropoffs[dropoffs['BOARDING_STOP'].astype(str).str[:-2].isin(viable_orig.get(s, []))].reset_index()

        arr_pax = []
        drop_pax = []
        for i in range(start_interval, start_interval + nr_intervals):
            temp_arr_pax = sum(arrivals[arrivals['bin_5'] == i]['mean'].tolist())
            arr_pax.append(temp_arr_pax/interval_length)
            temp_drop_pax = sum(dropoffs[dropoffs['bin_5'] == i]['mean'].tolist())
            drop_pax.append(temp_drop_pax/interval_length)
        arr_rates[s] = arr_pax
        drop_rates[s] = drop_pax
    dep_vol = {}
    prev_vol = [0] * nr_intervals
    for i in range(len(stops)):
        alight_fractions[stops[i]] = []
        for pv, d in zip(prev_vol, drop_rates[stops[i]]):
            try:
                alight_fractions[stops[i]].append(d/pv)
            except ZeroDivisionError:
                alight_fractions[stops[i]].append(0.0)
        dep_vol[stops[i]] = [round(pv + a - d, 2) for pv, a, d in zip(prev_vol, arr_rates[stops[i]], drop_rates[stops[i]])]
        prev_vol = dep_vol.get(stops[i])
    return arr_rates, alight_fractions


def read_scheduled_departures(path):
    sch_dep = []
    with open(path) as f:
        rf = csv.reader(f, delimiter=' ')
        for row in rf:
            sch_dep.append(int(row[0]) / 1e+09)
    return sch_dep


def get_historical_headway(pathname, start, end, dates, all_stops):
    whole_df = pd.read_csv(pathname)
    df = whole_df[whole_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['avl_sec'] % 86400 <= end]
    df = df[df['avl_sec'] % 86400 >= start]
    trip_ids = df['trip_id'].unique().tolist()
    all_stops = [int(s) for s in all_stops]
    hw_per_day = [{} for i in range(len(dates))]
    limit_headway = 16 * 60
    for i in range(len(dates)):
        df_temp = whole_df[whole_df['trip_id'].isin(trip_ids)]
        df_temp = df_temp[df_temp['event_time'].astype(str).str[:10] == dates[i]]
        for s in all_stops:
            df_temp1 = df_temp[df_temp['stop_id'] == s]
            df_temp1 = df_temp1.sort_values(by='avl_sec')
            times_temp = df_temp1['avl_sec'].tolist()
            hw_per_day[i][str(s)] = []
            for j in range(1, len(times_temp)):
                hw = times_temp[j] - times_temp[j-1]
                if hw <= limit_headway:
                    hw_per_day[i][str(s)].append(times_temp[j] - times_temp[j - 1])
    return hw_per_day


def plot_multiple_bar_charts(wta, wtc, lbls, pathname=False):
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


def plot_bar_chart(var, pathname=False):
    plt.bar(var.keys(), var.values())
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
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


def historical_headway(pathname_get, pathname_plot, start, end, dates, stops):
    # file is stop time from qing yi
    hh = get_historical_headway(pathname_get, start, end, dates, stops)
    path_hw = pathname_plot
    for i in range(len(dates)):
        plot_stop_headway(hh[i], path_hw.replace('!', dates[i]))
    return


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
