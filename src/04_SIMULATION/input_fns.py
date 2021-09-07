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
    reasonable_difference = 10*60

    df1 = df.sort_values(by='schd_sec')
    df1 = df1.drop_duplicates(subset='schd_trip_id')
    ordered_trips = df1['schd_trip_id'].tolist()
    # ordered_trip_stop_pattern = {}
    # for t in ordered_trips:
    #     for d in dates:
    #         single = whole_df[whole_df['trip_id'] == t]
    #         single = single[single['event_time'].astype(str).str[:10] == d]
    #         single = single.sort_values(by='stop_sequence')
    #         stop_sequence = single['stop_sequence'].tolist()
    #         res = all(i == j-1 for i, j in zip(stop_sequence, stop_sequence[1:]))
    #         if res:
    #             stop_ids = single['stop_id'].tolist()
    #             ordered_trip_stop_pattern[str(t)+'-'+str(d)] = stop_ids
    # with open('in/stop_pattern.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in ordered_trip_stop_pattern.items():
    #         writer.writerow([key, value])
    # df = df.sort_values(by='avl_sec')
    # df1.to_csv('in/ordered_dispatching.csv', index=False)
    # df.to_csv('in/trip_dispatching.csv', index=False)

    trip_ids = df['trip_id'].unique().tolist()
    # corrupted_trip_ids = []
    for d in dates:
        df_to_check = whole_df[whole_df['trip_id'].isin(trip_ids)]
        df_to_check = df_to_check[df_to_check['event_time'].astype(str).str[:10] == d]
        df_to_check = df_to_check.sort_values(by='avl_sec')
        df_to_check.to_csv('in/trips_'+str(d)+'.csv', index=False)
    for t in trip_ids:
        temp = route_stop_times[route_stop_times['trip_id'] == t]
        temp = temp.sort_values(by='stop_sequence')
        for d in dates:
            date_specific = temp[temp['event_time'].astype(str).str[:10] == d]
            # diff = date_specific['avl_sec'].mod(86400) - date_specific['schd_sec']
            # diff = diff.abs()
            # if diff.min() <= reasonable_difference:
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
                            lt = times[i+1] - times[i]
                            if lt > 0:
                                link_times[link][nr_bin].append(lt)
            # else:
            #     corrupted_trip_ids.append([t, d, date_specific['diff'].min(), diff.min()])
    # print(corrupted_trip_ids)
    # print(len(corrupted_trip_ids))
    # print(len(trip_ids) * len(dates))
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
    return all_stops, mean_link_times, stdev_link_times, nr_dpoints_link_times, ordered_trips


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


def get_dispatching_from_gtfs(pathname, ordered_trips):
    df = pd.read_csv(pathname)
    scheduled_departures = df[df['trip_id'].isin(ordered_trips)]['schd_sec'].tolist()
    return scheduled_departures


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
                        headway[s].append(hw)
                    else:
                        headway[s] = [hw]
    return headway


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
            plt.scatter(str(stop), h, color='r', s=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
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

