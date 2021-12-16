import csv
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
from scipy.stats import norm, lognorm
from fitter import Fitter


def get_interval(t, len_i_mins):
    # t is in seconds and len_i in minutes
    interval = int(t/(len_i_mins*60))
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
              visualize_data=False, tolerance_early_departure=1.5*60):
    link_times = {}
    link_times_true = {}
    stop_times_df = pd.read_csv(path_stop_times)

    df = stop_times_df[stop_times_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['schd_sec'] % 86400 <= end_time]
    df = df[df['schd_sec'] % 86400 >= start_time_extract]
    trip_ids_tt_extract = df['trip_id'].unique().tolist()

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
            avl_dep_sec = date_specific['avl_dep_sec'].tolist()
            stop_sequence = date_specific['stop_sequence'].tolist()
            if avl_sec:
                if stop_sequence[0] == 1:
                    if schd_sec[0] - (avl_dep_sec[0] % 86400) > tolerance_early_departure:
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
                            link_times_true[link] = [[] for i in range(nr_intervals)]
                        nr_bin = get_interval(avl_sec[i] % 86400, interval_length) - start_interval
                        if 0 <= nr_bin < nr_intervals:
                            lt = avl_sec[i+1] - avl_sec[i]
                            lt2 = avl_sec[i+1] - avl_dep_sec[i]
                            if lt > 0:
                                link_times[link][nr_bin].append(lt)
                                link_times_true[link][nr_bin].append(lt2)
    mean_link_times = {}
    mean_link_times_true = {}
    stdev_link_times = {}
    stdev_link_times_true = {}
    nr_dpoints_link_times = {}
    nr_dpoints_link_times_true = {}
    for link in link_times:
        mean_link_times[link] = []
        stdev_link_times[link] = []
        nr_dpoints_link_times[link] = []
        for b in link_times[link]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                b_array = np.array(b)
                b_array = remove_outliers(b_array)
                mean_link_times[link].append(round(b_array.mean(), 1))
                stdev_link_times[link].append(round(b_array.std(), 1))
                nr_dpoints_link_times[link].append(len(b_array))
    for link in link_times_true:
        mean_link_times_true[link] = []
        stdev_link_times_true[link] = []
        nr_dpoints_link_times_true[link] = []
        for b in link_times_true[link]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                b_array = np.array(b)
                b_array = remove_outliers(b_array)
                mean_link_times_true[link].append(round(b_array.mean(),1))
                stdev_link_times_true[link].append(round(b_array.std(),1))
                nr_dpoints_link_times_true[link].append(len(b_array))

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
            plt.close()
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

    # link_times_info = (mean_link_times, stdev_link_times, nr_dpoints_link_times)
    link_times_true_info = (mean_link_times_true, stdev_link_times_true, nr_dpoints_link_times_true)
    return all_stops, trip_ids_simulation, link_times_true_info


def get_demand(path, stops, nr_intervals, start_interval, new_nr_intervals, new_interval_length):
    arr_rates = {}
    drop_rates = {}
    alight_fractions = {}
    od_pairs = pd.read_csv(path)
    viable_dest = {}
    viable_orig = {}
    grouping = int(nr_intervals / new_nr_intervals)
    od_set = np.zeros(shape=(new_nr_intervals, len(stops), len(stops)))
    od_set[:] = np.nan

    for i in range(len(stops)):
        for j in range(i+1, len(stops)):
            pax_df = od_pairs[od_pairs['BOARDING_STOP'].astype(str).str[:-2] == stops[i]].reset_index()
            pax_df = pax_df[pax_df['INFERRED_ALIGHTING_GTFS_STOP'].astype(str).str[:-2] == stops[j]].reset_index()
            counter = 0
            for k in range(start_interval, start_interval + nr_intervals, grouping):
                temp_pax = 0
                for m in range(k, k + grouping):
                    mean_interval_pax = pax_df[pax_df['bin_5'] == m]['mean']
                    if not mean_interval_pax.empty:
                        temp_pax += float(mean_interval_pax)
                if temp_pax:
                    od_set[counter, i, j] = temp_pax * 60 / new_interval_length # pax per hr
                counter += 1

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
    return arr_rates, alight_fractions, drop_rates, dep_vol, od_set


def get_dispatching_from_gtfs(pathname, trip_ids_simulation):
    df = pd.read_csv(pathname)
    scheduled_departures = df[df['trip_id'].isin(trip_ids_simulation)]['schd_sec'].tolist()
    return scheduled_departures


def get_trip_times(stop_times_path, focus_trips, dates, start_time, end_time,
                   tolerance_early_departure=1.5*60):

    # TRIP TIMES ARE USED FOR CALIBRATION THEREFORE THEY ONLY USE THE FOCUS TRIPS FOR RESULTS ANALYSIS
    trip_times = []
    stop_times_df = pd.read_csv(stop_times_path)
    for t in focus_trips:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['event_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 67:
                    arrival_sec = df['avl_sec'].tolist()
                    dep_sec = df['avl_dep_sec'].tolist()
                    schd_sec = df['schd_sec'].tolist()
                    if schd_sec[0] - (dep_sec[0] % 86400) < tolerance_early_departure:
                        trip_times.append(arrival_sec[-1] - dep_sec[0])

    # THIS WE WILL USE TO VALIDATE THE OUTBOUND DIRECTION MODELING HENCE WE TAKE ALL DEPARTURES IN THE PERIOD OF STUDY
    departure_headway = []
    df = stop_times_df[stop_times_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['schd_sec'] % 86400 <= end_time]
    df = df[df['schd_sec'] % 86400 >= start_time]
    df = df.sort_values(by='schd_sec')
    ordered_trip_ids = df['trip_id'].unique().tolist()
    df = stop_times_df[stop_times_df['trip_id'].isin(ordered_trip_ids)]
    for d in dates:
        temp_df = df[df['event_time'].astype(str).str[:10] == d]
        temp_df = temp_df[temp_df['stop_id'] == 386]
        temp_df = temp_df.sort_values(by='schd_sec')
        avl_sec = temp_df['avl_dep_sec'].tolist()
        # trip_ids = temp_df['trip_id'].tolist()
        if avl_sec:
            avl_sec.sort()
            temp_dep_hw = [i - j for i, j in zip(avl_sec[1:], avl_sec[:-1])]
            departure_headway += temp_dep_hw
    departure_headway = remove_outliers(np.array(departure_headway)).tolist()
    nr_bins = 15
    plt.hist(departure_headway, ec='black', bins=nr_bins)
    # plt.xticks(range(-1000, 2200, 200), rotation='vertical')
    plt.xlabel('inbound departure headway (seconds)')
    plt.tight_layout()
    plt.savefig('in/vis/departure_headway_inbound.png')
    plt.close()
    # print(f'departure headway: \nmean:{np.mean(departure_headway)} \nmedian: {np.median(departure_headway)}\nstd: {np.std(departure_headway)}')
    return trip_times, departure_headway


def get_dwell_times(stop_times_path, focus_trips, stops, dates):
    dwell_times_mean = {}
    dwell_times_std = {}
    dwell_times_tot = []
    stop_times_df = pd.read_csv(stop_times_path)
    for t in focus_trips:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for s in stops[1:-1]:
            df = temp_df[temp_df['stop_id'] == int(s)]
            if not df.empty:
                dwell_times_mean[s] = df['dwell_time'].mean()
                dwell_times_std[s] = df['dwell_time'].std()
        for d in dates:
            df_date = temp_df[temp_df['event_time'].astype(str).str[:10] == d]
            df_date = df_date.sort_values(by='stop_sequence')
            stop_seq = df_date['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 67:
                    dwell_times = df_date['dwell_time'].tolist()
                    if len(dwell_times) == 67:
                        dwell_times_tot.append(sum(dwell_times[1:-1]))
    return dwell_times_mean, dwell_times_std, dwell_times_tot


def write_inbound_trajectories(stop_times_path, ordered_trips):
    stop_times_df = pd.read_csv(stop_times_path)
    stop_times_df = stop_times_df[stop_times_df['trip_id'].isin(ordered_trips)]
    stop_times_df = stop_times_df.sort_values(by=['stop_sequence', 'schd_sec'])
    stop_times_df.to_csv('in/vis/trajectories_inbound.csv', index=False)
    return


def get_load_profile(stop_times_path, focus_trips, stops):
    lp = {}
    stop_times_df = pd.read_csv(stop_times_path)
    stop_times_df = stop_times_df[stop_times_df['trip_id'].isin(focus_trips)]
    for s in stops:
        df = stop_times_df[stop_times_df['stop_id'] == int(s)]
        lp[s] = df['passenger_load'].mean()
    return lp


def get_outbound_travel_time(path_stop_times, start_time, end_time, dates, tolerance_early_dep=1*60):
    stop_times_df = pd.read_csv(path_stop_times)
    # we choose the outbound trips that matter: whose arrivals fall between the period of study 6:56-10:00
    df = stop_times_df[stop_times_df['stop_sequence'].isin([23, 63])]
    df = df[df['stop_id'] == 386]
    df = df[df['schd_sec'] <= end_time]
    df = df[df['schd_sec'] >= start_time]
    df = df.sort_values(by='schd_sec')
    ordered_arriving_trip_ids = df['trip_id'].unique().tolist()

    df_arrivals = df.drop_duplicates(subset='trip_id')
    scheduled_arrivals = df_arrivals['schd_sec'].tolist()

    df1 = stop_times_df[stop_times_df['stop_sequence'] == 63]
    df1 = df1[df1['stop_id'] == 386]
    df1 = df1[df1['schd_sec'] <= end_time]
    df1 = df1[df1['schd_sec'] >= start_time]
    trip_ids1 = df1['trip_id'].unique().tolist()

    df_deps1 = stop_times_df[stop_times_df['trip_id'].isin(trip_ids1)]
    df_deps1 = df_deps1[df_deps1['stop_sequence'] == 1]
    df_deps1 = df_deps1.sort_values(by='schd_sec')
    df_deps1 = df_deps1.drop_duplicates(subset='trip_id')
    sched_deps1 = df_deps1['schd_sec'].tolist()
    add_sched_dep_time = (datetime.strptime('7:33:30', '%H:%M:%S')-datetime(1900, 1, 1)).total_seconds()
    sched_deps1.append(int(add_sched_dep_time))

    df2 = stop_times_df[stop_times_df['stop_sequence'] == 23]
    df2 = df2[df2['stop_id'] == 386]
    df2 = df2[df2['schd_sec'] <= end_time]
    df2 = df2[df2['schd_sec'] >= start_time]
    trip_ids2 = df2['trip_id'].unique().tolist()

    df_deps2 = stop_times_df[stop_times_df['trip_id'].isin(trip_ids2)]
    df_deps2 = df_deps2[df_deps2['stop_sequence'] == 1]
    df_deps2 = df_deps2.sort_values(by='schd_sec')
    df_deps2 = df_deps2.drop_duplicates(subset='trip_id')
    sched_deps2 = df_deps2['schd_sec'].tolist()

    trip_times1 = []
    dep_delay1 = []
    for t in trip_ids1:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['event_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 63:
                    arrival_sec = df['avl_sec'].tolist()
                    dep_sec = df['avl_dep_sec'].tolist()
                    schd_sec = df['schd_sec'].tolist()
                    dep_delay = schd_sec[0] - (dep_sec[0] % 86400)
                    dep_delay1.append(-dep_delay)
                    if dep_delay < tolerance_early_dep:
                        trip_times1.append(arrival_sec[-1] - dep_sec[0])
    trip_times2 = []
    dep_delay2 = []
    for t in trip_ids2:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['event_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 23:
                    arrival_sec = df['avl_sec'].tolist()
                    dep_sec = df['avl_dep_sec'].tolist()
                    schd_sec = df['schd_sec'].tolist()
                    dep_delay = schd_sec[0] - (dep_sec[0] % 86400)
                    dep_delay2.append(-dep_delay)
                    if dep_delay < tolerance_early_dep:
                        trip_times2.append(arrival_sec[-1] - dep_sec[0])

    arrival_headway = []

    df = stop_times_df[stop_times_df['trip_id'].isin(ordered_arriving_trip_ids)]
    df = df.sort_values(by=['stop_sequence', 'schd_sec'])
    df.to_csv('in/vis/trajectories_outbound.csv', index=False)
    df_arrivals = df[df['stop_id'] == 386]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals.to_csv('in/vis/arrivals_outbound.csv', index=False)
    for d in dates:
        temp_df = df[df['event_time'].astype(str).str[:10] == d]
        temp_df = temp_df[temp_df['stop_id'] == 386]
        temp_df = temp_df.sort_values(by='schd_sec')
        # trip_ids = temp_df['trip_id'].tolist()
        avl_sec = temp_df['avl_sec'].tolist()
        if avl_sec:
            avl_sec.sort()
            temp_arr_hw = [i - j for i, j in zip(avl_sec[1:], avl_sec[:-1])]
            arrival_headway += temp_arr_hw

    dep_delay1 = np.array(dep_delay1)
    dep_delay1 = dep_delay1[dep_delay1 >= 0].tolist()
    dep_delay2 = np.array(dep_delay2)
    dep_delay2 = dep_delay2[dep_delay2 >= 0].tolist()
    dep_delay1 = remove_outliers(np.array(dep_delay1)).tolist()
    dep_delay2 = remove_outliers(np.array(dep_delay2)).tolist()
    arrival_headway = remove_outliers(np.array(arrival_headway)).tolist()
    trip_times1 = remove_outliers(np.array(trip_times1)).tolist()
    trip_times2 = remove_outliers(np.array(trip_times2)).tolist()

    sns.kdeplot(trip_times1)
    plt.xlabel('total trip time (seconds)')
    plt.savefig('in/vis/trip_time_outbound(long).png')
    plt.close()
    sns.kdeplot(trip_times2)
    plt.xlabel('total trip time (seconds)')
    plt.savefig('in/vis/trip_time_outbound(short).png')
    plt.close()
    plt.hist(arrival_headway, ec='black', bins=15)
    plt.xlabel('outbound arrival headway (seconds)')
    plt.tight_layout()
    plt.savefig('in/vis/arrival_hw_outbound.png')
    plt.close()
    plt.hist(dep_delay1, ec='black', bins=15)
    plt.xlabel('outbound (long) departure delay (seconds)')
    plt.tight_layout()
    plt.savefig('in/vis/dep_delay_outbound(long).png')
    plt.close()
    plt.hist(dep_delay2, ec='black', bins=15)
    plt.xlabel('outbound (short) departure delay (seconds)')
    plt.tight_layout()
    plt.savefig('in/vis/dep_delay_outbound(short).png')
    plt.close()

    dep_delay1_params = lognorm.fit(dep_delay1, loc=0)

    trip_time1_distribution = Fitter(trip_times1, distributions=['norm'])
    trip_time1_distribution.fit()
    # trip_time1_distribution.summary()
    # plt.show()
    trip_time1_params = trip_time1_distribution.fitted_param['norm']

    dep_delay2_params = lognorm.fit(dep_delay2, loc=0)

    trip_times2_distribution = Fitter(trip_times2, distributions=['norm'])
    trip_times2_distribution.fit()
    # trip_times2_distribution.summary()
    # plt.show()
    trip_times2_params = trip_times2_distribution.fitted_param['norm']
    # print(f'outbound long trip times: \nmean:{np.mean(trip_times1)} \nmedian: {np.median(trip_times1)}\nstd: {np.std(trip_times1)}')
    # print(f'outbound short trip times: \nmean:{np.mean(trip_times2)} \nmedian: {np.median(trip_times2)}\nstd: {np.std(trip_times2)}')
    # print(f'arrival headway: \nmean:{np.mean(arrival_headway)} \nmedian: {np.median(arrival_headway)}\nstd: {np.std(arrival_headway)}')
    # print(f'dep delay (long): \nmean:{np.mean(dep_delay1)} \nmedian: {np.median(dep_delay1)}\nstd: {np.std(dep_delay1)}')
    # print(f'dep delay (short): \nmean:{np.mean(dep_delay2)} \nmedian: {np.median(dep_delay2)}\nstd: {np.std(dep_delay2)}')
    return sched_deps1, sched_deps2, scheduled_arrivals, dep_delay1_params, trip_time1_params, dep_delay2_params, trip_times2_params, arrival_headway


def get_scheduled_bus_availability(path_stop_times, dates, start_time, end_time):
    stop_times_df = pd.read_csv(path_stop_times)
    date = dates[2]
    stop_times_df = stop_times_df[stop_times_df['event_time'].astype(str).str[:10] == date]

    df_outbound = stop_times_df[stop_times_df['stop_sequence'].isin([23, 63])]
    df_outbound = df_outbound[df_outbound['stop_id'] == 386]
    df_outbound = df_outbound[df_outbound['schd_sec'] >= start_time]
    df_outbound = df_outbound[df_outbound['schd_sec'] <= end_time]
    df_outbound = df_outbound.sort_values(by='avl_sec')
    actual_arrivals = df_outbound['avl_sec'] % 86400
    actual_arrivals = actual_arrivals.tolist()
    df_outbound = df_outbound.sort_values(by='schd_sec')
    scheduled_arrivals = df_outbound['schd_sec'].tolist()

    df_inbound = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df_inbound = df_inbound[df_inbound['stop_id'] == 386]
    df_inbound = df_inbound[df_inbound['schd_sec'] >= start_time]
    df_inbound = df_inbound[df_inbound['schd_sec'] <= end_time]
    df_inbound = df_inbound.sort_values(by='avl_dep_sec')
    actual_departures = df_inbound['avl_dep_sec'] % 86400
    actual_departures = actual_departures.tolist()
    df_inbound = df_inbound.sort_values(by='schd_sec')
    scheduled_departures = df_inbound['schd_sec'].tolist()

    plt.plot(scheduled_departures, [i for i in range(1, len(scheduled_departures)+1)], '--', label='scheduled_departures', color='red')
    plt.plot(scheduled_arrivals, [i for i in range(1, len(scheduled_arrivals)+1)], '--', label='scheduled arrivals', color='green')
    plt.plot(actual_arrivals, [i for i in range(1, len(actual_arrivals)+1)], label='avl arrivals')
    plt.plot(actual_departures, [i for i in range(1, len(actual_departures) + 1)], label='avl departures')
    plt.xlabel('time (seconds)')
    plt.ylabel('number of buses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'in/vis/bus_availability_{date}.png')
    plt.close()
    return
