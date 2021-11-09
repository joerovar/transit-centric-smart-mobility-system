import csv
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy


def get_interval(t, len_i):
    # t is in seconds and len_i in minutes
    interval = int(t/(len_i*60))
    return interval


def remove_outliers(data):
    if data.any():
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data >= lower_bound) & (data <= upper_bound)]
    return data


def get_route(path_stop_times, start_time_extract, end_time, nr_intervals, start_interval, interval_length,
              dates, trip_choice, pathname_dispatching, pathname_sorted_trips, pathname_stop_pattern, start_time,
              focus_start_time, focus_end_time,
              visualize_data=True, tolerance_early_departure=1.5*60):
    link_times = {}
    stop_times_df = pd.read_csv(path_stop_times)

    df = stop_times_df[stop_times_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['avl_sec'] % 86400 <= end_time]
    df = df[df['avl_sec'] % 86400 >= start_time_extract]
    trip_ids_tt_extract = df['trip_id'].unique().tolist()

    # df_faulty_dispatch_t = df[df['avl_sec'] % 86400 + 120 < df['schd_sec']]
    # df_faulty_dispatch_t.to_csv('in/vis/faulty_dispatching_times.csv', index=False)

    df_dispatching = df[df['schd_sec'] % 86400 >= start_time]
    df_dispatching = df_dispatching.sort_values(by='schd_sec')
    df_dispatching = df_dispatching.drop_duplicates(subset='schd_trip_id')
    trip_ids_simulation = df_dispatching['schd_trip_id'].tolist()

    df_dispatching.to_csv(pathname_dispatching, index=False)

    for t in trip_ids_tt_extract:
        temp = stop_times_df[stop_times_df['trip_id'] == t]
        temp = temp.sort_values(by='stop_sequence')
        for d in dates:
            date_specific = temp[temp['event_time'].astype(str).str[:10] == d]
            schd_sec = date_specific['schd_sec'].tolist()
            stop_id = date_specific['stop_id'].astype(str).tolist()
            avl_sec = date_specific['avl_sec'].tolist()
            stop_sequence = date_specific['stop_sequence'].tolist()
            if avl_sec:
                if stop_sequence[0] == 1:
                    if schd_sec[0] - (avl_sec[0] % 86400) > tolerance_early_departure:
                        schd_sec.pop(0)
                        stop_id.pop(0)
                        avl_sec.pop(0)
                        stop_sequence.pop(0)
                for i in range(len(stop_id)-1):
                    if stop_sequence[i] == stop_sequence[i + 1] - 1:
                        link = stop_id[i]+'-'+stop_id[i+1]
                        exists = link in link_times
                        if not exists:
                            link_times[link] = [[] for i in range(nr_intervals)]
                        nr_bin = get_interval(avl_sec[i] % 86400, interval_length) - start_interval
                        if 0 <= nr_bin < nr_intervals:
                            lt = avl_sec[i+1] - avl_sec[i]
                            if lt > 0:
                                link_times[link][nr_bin].append(lt)
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

    df_forstops = stop_times_df[stop_times_df['trip_id'] == trip_choice]
    df_forstops = df_forstops[df_forstops['event_time'].astype(str).str[:10] == dates[0]]
    df_forstops = df_forstops.sort_values(by='stop_sequence')
    all_stops = df_forstops['stop_id'].astype(str).tolist()

    ordered_trip_stop_pattern = {}

    if visualize_data:
        # daily trips, optional to visualize future trips
        for d in dates:
            df_day_trips = stop_times_df[stop_times_df['trip_id'].isin(trip_ids_tt_extract)]
            df_day_trips = df_day_trips[df_day_trips['event_time'].astype(str).str[:10] == d]
            df_day_trips = df_day_trips.sort_values(by='avl_sec')
            df_day_trips.to_csv(pathname_sorted_trips + str(d) + '.csv', index=False)
            df_day_trips['avl_sec'] = df_day_trips['avl_sec'] % 86400
            df_plot = df_day_trips.loc[(df_day_trips['avl_sec'] >= focus_start_time) &
                                       (df_day_trips['avl_sec'] <= focus_end_time)]

            fig, ax = plt.subplots()
            df_plot.reset_index().groupby(['trip_id']).plot(x='avl_sec', y='stop_sequence', ax=ax,
                                                            legend=False)
            plt.savefig('in/vis/historical_trajectories' + d + '.png')

        # stop pattern to spot differences
        for t in trip_ids_simulation:
            for d in dates:
                single = stop_times_df[stop_times_df['trip_id'] == t]
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

    return all_stops, mean_link_times, stdev_link_times, nr_dpoints_link_times, trip_ids_simulation


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

    dep_vol = {}
    prev_vol = [0] * new_nr_intervals
    for i in range(len(stops)):
        j = 0
        alight_fractions[stops[i]] = []
        for pv, a, d in zip(prev_vol, arr_rates[stops[i]], drop_rates[stops[i]]):
            af = d / pv if pv else 0
            alight_fractions[stops[i]].append(af)
            prev_vol[j] = pv + a - d
            j += 1
        dep_vol[stops[i]] = deepcopy(prev_vol)
    return arr_rates, alight_fractions, drop_rates, dep_vol


def read_scheduled_departures(path):
    sch_dep = []
    with open(path) as f:
        rf = csv.reader(f, delimiter=' ')
        for row in rf:
            sch_dep.append(int(row[0]) / 1e+09)
    return sch_dep


def get_dispatching_from_gtfs(pathname, trip_ids_simulation):
    df = pd.read_csv(pathname)
    scheduled_departures = df[df['trip_id'].isin(trip_ids_simulation)]['schd_sec'].tolist()
    return scheduled_departures


def get_pax_per_trip(rates, start_time, end_time, start_interval, interval_length, avg_headway):
    # time in HOURS
    single_rate = {}
    for s in rates:
        # get interval has unit requirements as t in seconds and interval length in minutes
        inter1 = get_interval(start_time*3600, interval_length*60)
        inter2 = get_interval(end_time*3600, interval_length*60)
        numer = 0
        denom = 0
        for i in range(inter1, inter2+1):
            idx = i - start_interval
            if i == inter1:
                length_temp = (i+1) * interval_length - start_time
                numer += length_temp * rates[s][idx]
                denom += length_temp
            elif i == inter2:
                length_temp = end_time - i * interval_length
                numer += length_temp * rates[s][idx]
                denom += length_temp
            else:
                numer += interval_length * rates[s][idx]
                denom += interval_length
            single_rate[s] = avg_headway * numer/denom
    return single_rate
