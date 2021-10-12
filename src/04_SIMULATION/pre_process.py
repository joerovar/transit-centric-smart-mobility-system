import csv
import numpy as np
import pandas as pd
import warnings


def get_interval(t, len_i):
    interval = int(t/(len_i*60))
    return interval


def remove_outliers(data, m=2):
    dev = np.abs(data-np.median(data))
    median_dev = np.median(dev)
    s = dev/median_dev if median_dev else 0
    return data[s < m]


def get_route(path_stop_times, start_time, end_time, nr_intervals, start_interval, interval_length, dates, trip_choice, pathname_dispatching, pathname_sorted_trips, pathname_stop_pattern):
    link_times = {}
    route_times = []
    route_stop_times = pd.read_csv(path_stop_times)
    whole_df = pd.read_csv(path_stop_times)
    df = whole_df[whole_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['avl_sec'] % 86400 <= end_time]
    df = df[df['avl_sec'] % 86400 >= start_time]
    df1 = df.sort_values(by='schd_sec')
    df1 = df1.drop_duplicates(subset='schd_trip_id')
    ordered_trips = df1['schd_trip_id'].tolist()
    ordered_trip_stop_pattern = {}
    # dummy = []
    for t in ordered_trips:
        for d in dates:
            single = whole_df[whole_df['trip_id'] == t]
            single = single[single['event_time'].astype(str).str[:10] == d]
            single = single.sort_values(by='stop_sequence')
            stop_sequence = single['stop_sequence'].tolist()
            res = all(i == j-1 for i, j in zip(stop_sequence, stop_sequence[1:]))
            if res:
                stop_ids = single['stop_id'].tolist()
                ordered_trip_stop_pattern[str(t)+'-'+str(d)] = stop_ids
    with open(pathname_stop_pattern, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in ordered_trip_stop_pattern.items():
            writer.writerow([key, value])
    df1.to_csv(pathname_dispatching, index=False)

    trip_ids = df['trip_id'].unique().tolist()
    for d in dates:
        df_to_check = whole_df[whole_df['trip_id'].isin(trip_ids)]
        df_to_check = df_to_check[df_to_check['event_time'].astype(str).str[:10] == d]
        df_to_check = df_to_check.sort_values(by='avl_sec')
        df_to_check.to_csv(pathname_sorted_trips+str(d)+'.csv', index=False)
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
                            lt = times[i+1] - times[i]
                            if lt > 0:
                                link_times[link][nr_bin].append(lt)
                                # if link == '386-388':
                                #     dummy.append([t, d, lt])
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
                b_array = np.array(b)
                b_array = remove_outliers(b_array)
                mean_link_times[link].append(b_array.mean())
                stdev_link_times[link].append(b_array.std())
                nr_dpoints_link_times[link].append(len(b_array))
    df_forstops = route_stop_times[route_stop_times['trip_id'] == trip_choice]
    df_forstops = df_forstops[df_forstops['event_time'].astype(str).str[:10] == dates[0]]
    df_forstops = df_forstops.sort_values(by='stop_sequence')
    all_stops = df_forstops['stop_id'].astype(str).tolist()

    return all_stops, mean_link_times, stdev_link_times, nr_dpoints_link_times, ordered_trips


def get_demand(path, stops, nr_intervals, start_interval, new_nr_intervals, new_interval_length):
    arr_rates = {}
    drop_rates = {}
    alight_fractions = {}
    # first we turn it into an OD matrix
    od_pairs = pd.read_csv(path)
    viable_dest = {}
    viable_orig = {}
    grouping = int(nr_intervals / new_nr_intervals)
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
        for i in range(start_interval, start_interval + nr_intervals, grouping):
            temp_arr_pax = 0
            temp_drop_pax = 0
            for j in range(i, i+grouping):
                temp_arr_pax += sum(arrivals[arrivals['bin_5'] == j]['mean'].tolist())
                temp_drop_pax += sum(dropoffs[dropoffs['bin_5'] == j]['mean'].tolist())
            arr_pax.append(float(temp_arr_pax*60 / new_interval_length)) # convert each rate to pax/hr, more intuitive
            drop_pax.append(float(temp_drop_pax*60 / new_interval_length))
        arr_rates[s] = arr_pax
        drop_rates[s] = drop_pax

    # if new_nr_intervals:
    #     for s in arr_rates:
    #         temp_ar = arr_rates[s]
    #         temp_af = drop_rates[s]
    #         n = grouping
    #         new_temp_ar = [np.array(temp_ar[i:i+n]).mean() for i in range(0, len(temp_ar), n)]
    #         new_temp_af = [np.array(temp_af[i:i+n]).mean() for i in range(0, len(temp_af), n)]
    #         arr_rates[s] = new_temp_ar
    #         alight_fractions[s] = new_temp_af
    dep_vol = {}
    prev_vol = [0] * new_nr_intervals
    for i in range(len(stops)):
        j = 0
        alight_fractions[stops[i]] = []
        for pv, a, d in zip(prev_vol, arr_rates[stops[i]], drop_rates[stops[i]]):
            print(d, pv)
            af = d / pv if pv else 0
            alight_fractions[stops[i]].append(af)
            prev_vol[j] = pv + a - d
            j += 1
        print(alight_fractions[stops[i]])
        print([stops[i], prev_vol])
        dep_vol[stops[i]] = prev_vol
    print(f'dep volume {dep_vol}')
    print(drop_rates)
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

