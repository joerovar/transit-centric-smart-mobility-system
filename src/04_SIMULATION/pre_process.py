from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm


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


def get_route(path_stop_times, start_time_sec, end_time_sec, nr_intervals, start_interval, interval_length,
              dates, trip_choice, tolerance_early_departure=1.5*60):
    link_times = {}
    link_times_true = {}
    stop_times_df = pd.read_csv(path_stop_times)

    df = stop_times_df[stop_times_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['schd_sec'] % 86400 <= end_time_sec]
    df = df[df['schd_sec'] % 86400 >= start_time_sec]
    trip_ids = df['trip_id'].unique().tolist()

    df_dispatching = df.sort_values(by='schd_sec')
    df_dispatching = df_dispatching.drop_duplicates(subset='trip_id')
    ordered_trip_ids = df_dispatching['trip_id'].tolist()
    scheduled_departures = df_dispatching[df_dispatching['trip_id'].isin(ordered_trip_ids)]['schd_sec'].tolist()

    df_arrivals = stop_times_df[stop_times_df['trip_id'].isin(ordered_trip_ids)]
    df_arrivals = df_arrivals[df_arrivals['stop_sequence'] == 67]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals = df_arrivals.drop_duplicates(subset='trip_id')
    sched_arrivals = df_arrivals['schd_sec'].tolist()

    for t in trip_ids:
        temp = stop_times_df[stop_times_df['trip_id'] == t]
        temp = temp.sort_values(by='stop_sequence')
        for d in dates:
            date_specific = temp[temp['avl_arr_time'].astype(str).str[:10] == d]
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
                        if link not in link_times:
                            link_times[link] = [[] for i in range(nr_intervals)]
                            link_times_true[link] = [[] for i in range(nr_intervals)]
                        nr_bin = get_interval(avl_sec[i] % 86400, interval_length) - start_interval
                        if 0 <= nr_bin < nr_intervals:
                            lt = avl_sec[i+1] - avl_sec[i]
                            lt2 = avl_sec[i+1] - avl_dep_sec[i]
                            if lt > 0:
                                link_times[link][nr_bin].append(lt)
                                link_times_true[link][nr_bin].append(lt2)

    mean_link_times_true = {}
    stdev_link_times_true = {}
    nr_dpoints_link_times_true = {}

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
    df_forstops = df_forstops[df_forstops['avl_arr_time'].astype(str).str[:10] == dates[0]]
    df_forstops = df_forstops.sort_values(by='stop_sequence')
    all_stops = df_forstops['stop_id'].astype(str).tolist()

    # mean_link_times = {}
    # stdev_link_times = {}
    # nr_dpoints_link_times = {}
    # for link in link_times:
    #     mean_link_times[link] = []
    #     stdev_link_times[link] = []
    #     nr_dpoints_link_times[link] = []
    #     for b in link_times[link]:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", category=RuntimeWarning)
    #             b_array = np.array(b)
    #             b_array = remove_outliers(b_array)
    #             mean_link_times[link].append(round(b_array.mean(), 1))
    #             stdev_link_times[link].append(round(b_array.std(), 1))
    #             nr_dpoints_link_times[link].append(len(b_array))
    # link_times_info = (mean_link_times, stdev_link_times, nr_dpoints_link_times)

    link_times_true_info = (mean_link_times_true, stdev_link_times_true, nr_dpoints_link_times_true)
    return all_stops, ordered_trip_ids, link_times_true_info, scheduled_departures, sched_arrivals


def get_demand(path_odt, path_stop_times, stops, input_start_interval, input_end_interval, start_interval,
               end_interval, proportion_intervals, interval_length, dates):
    arr_rates = np.zeros(shape=(end_interval - start_interval, len(stops)))
    drop_rates = np.zeros(shape=(end_interval - start_interval, len(stops)))
    apc_df = pd.read_csv(path_stop_times)
    inbound_df = apc_df[apc_df['stop_id'] == int(stops[0])]
    inbound_df = inbound_df[inbound_df['stop_sequence'] == 1]
    inbound_trips = inbound_df['trip_id'].unique().tolist()
    apc_df = apc_df[apc_df['trip_id'].isin(inbound_trips)]
    for i in range(len(stops)):
        temp_df = apc_df[apc_df['stop_id'] == int(stops[i])]
        for j in range(start_interval, end_interval):
            t_edge0 = j * interval_length * 60
            t_edge1 = (j + 1) * interval_length * 60
            pax_df = temp_df[temp_df['avl_dep_sec'] <= t_edge1]
            pax_df = pax_df[pax_df['avl_dep_sec'] >= t_edge0]
            if i < len(stops) - 1:
                ons_rate = (pax_df['ron'].sum() + pax_df['fon'].sum()) * 60 / interval_length
                arr_rates[j - start_interval, i] = ons_rate
            if i:
                offs_rate = (pax_df['roff'].sum() + pax_df['foff'].sum()) * 60 / interval_length
                drop_rates[j - start_interval, i] = offs_rate
    odt_df = pd.read_csv(path_odt)
    input_interval_groups = []
    for i in range(input_start_interval, input_end_interval, proportion_intervals):
        input_interval_groups.append([j for j in range(i, i + proportion_intervals)])
    od_set = np.zeros(shape=(end_interval - start_interval, len(stops), len(stops)))
    od_set[:] = np.nan
    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            temp_df = odt_df[odt_df['BOARDING_STOP'] == float(stops[i])]
            temp_df = temp_df[temp_df['INFERRED_ALIGHTING_GTFS_STOP'] == float(stops[j])]
            for g in input_interval_groups:
                pax_df = temp_df[temp_df['bin_5'].isin(g)]
                pax = pax_df['mean'].sum()
                if pax:
                    time_idx = input_interval_groups.index(g)
                    od_set[time_idx, i, j] = pax * 60 / interval_length
    od_scaled_set = np.array(od_set)
    for i in range(od_set.shape[0]):
        od_scaled_set[i] = biproportional_fitting(od_set[i], arr_rates[i], drop_rates[i])
    ridership_non_scaled = np.nansum(od_set, axis=(1, -1))
    ridership_scaled = np.nansum(od_scaled_set, axis=(1, -1))
    ridership_apc = np.nansum(arr_rates, axis=-1)
    return arr_rates, drop_rates, od_scaled_set


def biproportional_fitting(od, target_ons, target_offs):
    balance_target = np.sum(target_ons) / np.sum(target_offs)
    balanced_target_offs = target_offs * balance_target
    for i in range(15):
        # balance rows
        actual_ons = np.nansum(od, axis=1)
        factor_ons = np.divide(target_ons, actual_ons, out=np.zeros_like(target_ons), where=actual_ons != 0)
        od = od * factor_ons[:, np.newaxis]

        # balance columns
        actual_offs = np.nansum(od, axis=0)
        factor_offs = np.divide(balanced_target_offs, actual_offs, out=np.zeros_like(target_offs), where=actual_offs != 0)
        od = od * factor_offs

        # to check for tolerance we first assign 1.0 to totals of zero which cannot be changed by the method
        factor_ons[actual_ons == 0] = 1.0
        factor_offs[actual_offs == 0] = 1.0
    scaled_od_set = np.array(od)
    return scaled_od_set


def get_outbound_travel_time(path_stop_times, start_time, end_time, dates, nr_intervals,
                             start_interval, interval_length, tolerance_early_dep=1 * 60):
    stop_times_df = pd.read_csv(path_stop_times)

    df1 = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df1 = df1[df1['stop_id'] == 8613]
    df1 = df1[df1['schd_sec'] <= end_time]
    df1 = df1[df1['schd_sec'] >= start_time]
    trip_ids1 = df1['trip_id'].unique().tolist()

    df_deps1 = stop_times_df[stop_times_df['trip_id'].isin(trip_ids1)]
    df_deps1 = df_deps1[df_deps1['stop_sequence'] == 1]
    df_deps1 = df_deps1.sort_values(by='schd_sec')
    df_deps1 = df_deps1.drop_duplicates(subset='trip_id')
    sched_deps1 = df_deps1['schd_sec'].tolist()
    add_sched_dep_time = (datetime.strptime('7:33:30', '%H:%M:%S') - datetime(1900, 1, 1)).total_seconds()
    sched_deps1.append(int(add_sched_dep_time))

    df2 = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df2 = df2[df2['stop_id'] == 15136]
    df2 = df2[df2['schd_sec'] <= end_time]
    df2 = df2[df2['schd_sec'] >= start_time]
    trip_ids2 = df2['trip_id'].unique().tolist()

    df_deps2 = stop_times_df[stop_times_df['trip_id'].isin(trip_ids2)]
    df_deps2 = df_deps2[df_deps2['stop_sequence'] == 1]
    df_deps2 = df_deps2.sort_values(by='schd_sec')
    df_deps2 = df_deps2.drop_duplicates(subset='trip_id')
    sched_deps2 = df_deps2['schd_sec'].tolist()

    all_trip_ids = trip_ids1 + trip_ids2
    df_arrivals = stop_times_df[stop_times_df['trip_id'].isin(all_trip_ids)]
    df_arrivals = df_arrivals[df_arrivals['stop_sequence'].isin([23, 63])]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals = df_arrivals.drop_duplicates(subset='trip_id')
    df_arrivals = df_arrivals[df_arrivals['schd_sec'] <= end_time]
    ordered_arriving_trip_ids = df_arrivals['trip_id'].tolist()
    sched_arrivals = df_arrivals['schd_sec'].tolist()

    trip_times1 = [[] for _ in range(nr_intervals)]
    deadhead_times = [[] for _ in range(nr_intervals)]
    stop_seq_terminal2 = 41
    dep_delay1 = []
    for t in trip_ids1:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
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
                        idx = get_interval(schd_sec[0], interval_length) - start_interval
                        trip_times1[idx].append(arrival_sec[-1] - dep_sec[0])
                if stop_seq[0] == 1 and stop_seq_terminal2 in stop_seq:
                    deadhead_df = df[df['stop_sequence'] <= stop_seq_terminal2]
                    deadhead_dep_t = deadhead_df['avl_dep_sec'].tolist()[0]
                    schd_dep_t = deadhead_df['schd_sec'].tolist()[0]
                    deadhead_arr_t = deadhead_df['avl_sec'].tolist()[-1]
                    if schd_dep_t - (deadhead_dep_t % 86400) < tolerance_early_dep:
                        # we take out the departure terminal because of faulty dwell time measurements
                        deadhead_df = deadhead_df[deadhead_df['stop_sequence'] > 1]
                        dwell_times = deadhead_df['dwell_time'].sum()
                        deadhead_time = (deadhead_arr_t - deadhead_dep_t) - dwell_times
                        idx = get_interval(schd_dep_t, interval_length) - start_interval
                        deadhead_times[idx].append(deadhead_time)
    trip_times2 = [[] for _ in range(nr_intervals)]
    dep_delay2 = []
    for t in trip_ids2:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
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
                        idx = get_interval(schd_sec[0], interval_length) - start_interval
                        trip_times2[idx].append(arrival_sec[-1] - dep_sec[0])
    arrival_headway = []
    df = stop_times_df[stop_times_df['trip_id'].isin(all_trip_ids)]
    df = df.sort_values(by=['stop_sequence', 'schd_sec'])
    df.to_csv('in/vis/trajectories_outbound.csv', index=False)
    df = stop_times_df[stop_times_df['trip_id'].isin(ordered_arriving_trip_ids)]
    df = df.sort_values(by=['stop_sequence', 'schd_sec'])
    df_arrivals = df[df['stop_id'] == 386]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals.to_csv('in/vis/arrivals_outbound.csv', index=False)
    for d in dates:
        temp_df = df[df['avl_arr_time'].astype(str).str[:10] == d]
        temp_df = temp_df[temp_df['stop_id'] == 386]
        temp_df = temp_df.sort_values(by='schd_sec')
        # trip_ids = temp_df['trip_id'].tolist()
        avl_sec = temp_df['avl_sec'].tolist()
        if avl_sec:
            avl_sec.sort()
            temp_arr_hw = [i - j for i, j in zip(avl_sec[1:], avl_sec[:-1])]
            arrival_headway += temp_arr_hw

    trip_times1_params = []
    trip_times2_params = []
    deadhead_times_params = []
    for i in range(nr_intervals):
        trip_times1[i] = remove_outliers(np.array(trip_times1[i])).tolist()
        trip_times2[i] = remove_outliers(np.array(trip_times2[i])).tolist()
        deadhead_times[i] = remove_outliers(np.array(deadhead_times[i])).tolist()
        trip_times1_params.append(norm.fit(trip_times1[i]))
        trip_times2_params.append(norm.fit(trip_times2[i]))
        deadhead_times_params.append(norm.fit(deadhead_times[i]))
    # dep_delay1 = np.array(dep_delay1)
    # dep_delay1 = dep_delay1[dep_delay1 >= 0].tolist()
    # dep_delay2 = np.array(dep_delay2)
    # dep_delay2 = dep_delay2[dep_delay2 >= 0].tolist()
    # dep_delay1 = remove_outliers(np.array(dep_delay1)).tolist()
    # dep_delay2 = remove_outliers(np.array(dep_delay2)).tolist()
    # arrival_headway = remove_outliers(np.array(arrival_headway)).tolist()
    # sns.kdeplot(trip_times1)
    # plt.xlabel('total trip time (seconds)')
    # plt.savefig('in/vis/trip_time_outbound(long).png')
    # plt.close()
    # sns.kdeplot(trip_times2)
    # plt.xlabel('total trip time (seconds)')
    # plt.savefig('in/vis/trip_time_outbound(short).png')
    # plt.close()
    # plt.hist(arrival_headway, ec='black', bins=15)
    # plt.xlabel('outbound arrival headway (seconds)')
    # plt.tight_layout()
    # plt.savefig('in/vis/arrival_hw_outbound.png')
    # plt.close()
    # plt.hist(dep_delay1, ec='black', bins=15)
    # plt.xlabel('outbound (long) departure delay (seconds)')
    # plt.tight_layout()
    # plt.savefig('in/vis/dep_delay_outbound(long).png')
    # plt.close()
    # plt.hist(dep_delay2, ec='black', bins=15)
    # plt.xlabel('outbound (short) departure delay (seconds)')
    # plt.tight_layout()
    # plt.savefig('in/vis/dep_delay_outbound(short).png')
    # plt.close()

    # dep_delay1_params = lognorm.fit(dep_delay1, loc=0)
    # dep_delay2_params = lognorm.fit(dep_delay2, loc=0)

    return sched_deps1, sched_deps2, sched_arrivals, trip_times1_params, trip_times2_params, deadhead_times_params


def get_trip_times(stop_times_path, focus_trips, dates, start_time, end_time,
                   tolerance_early_departure=1.5*60):

    # TRIP TIMES ARE USED FOR CALIBRATION THEREFORE THEY ONLY USE THE FOCUS TRIPS FOR RESULTS ANALYSIS
    trip_times = []
    stop_times_df = pd.read_csv(stop_times_path)
    for t in focus_trips:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
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
        temp_df = df[df['avl_arr_time'].astype(str).str[:10] == d]
        temp_df = temp_df[temp_df['stop_id'] == 386]
        temp_df = temp_df.sort_values(by='schd_sec')
        avl_sec = temp_df['avl_dep_sec'].tolist()
        if avl_sec:
            avl_sec.sort()
            temp_dep_hw = [i - j for i, j in zip(avl_sec[1:], avl_sec[:-1])]
            departure_headway += temp_dep_hw
    trip_times = remove_outliers(np.array(trip_times)).tolist()
    # departure_headway = remove_outliers(np.array(departure_headway)).tolist()
    # nr_bins = 15
    # plt.hist(departure_headway, ec='black', bins=nr_bins)
    # plt.xlabel('inbound departure headway (seconds)')
    # plt.tight_layout()
    # plt.savefig('in/vis/departure_headway_inbound.png')
    # plt.close()
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
            df_date = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
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


def get_scheduled_bus_availability(path_stop_times, dates, start_time, end_time):
    stop_times_df = pd.read_csv(path_stop_times)
    date = dates[2]
    stop_times_df = stop_times_df[stop_times_df['avl_arr_time'].astype(str).str[:10] == date]

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


def bus_availability(sched_arr, sched_dep, actual_arr, iden):
    actual_dep = [0] * len(sched_dep)
    for n in range(len(sched_dep)):
        actual_dep[n] = max(sched_dep[n], actual_arr[n])
    plt.plot(sched_arr, np.arange(1, len(sched_arr)+1),'--', label='scheduled arrivals', color='green', alpha=0.5)
    plt.plot(sched_dep, np.arange(1, len(sched_dep) + 1),'--', label='scheduled departures', color='red', alpha=0.5)
    plt.plot(actual_arr, np.arange(1, len(actual_arr) + 1),label='actual arrivals', color='green')
    plt.plot(actual_dep, np.arange(1, len(actual_dep) + 1),label='actual departures', color='red')
    plt.legend()
    plt.savefig('in/vis/sample_bus_availability' + iden + '.png')

    plt.close()
    actual_dep_hw = [j - k for j, k in zip(actual_dep[1:], actual_dep[:-1])]
    return actual_dep_hw


def simple_outbound_modeling(dep_delay1_params, dep_delay2_params, trip_time1_params, trip_time2_params,
                             sched_deps_out1, sched_deps_out2, sched_arr_out, sched_dep_in, observed_hw,
                             dep_hw_in):
    # # for generation of graphs you want:
    # dep_delay1_params = load('in/xtr/rt_20-2019-09/dep_delay1_params.pkl')
    # # print(dep_delay1_params)
    # trip_time1_params = load('in/xtr/rt_20-2019-09/trip_time1_params.pkl')
    # dep_delay2_params = load('in/xtr/rt_20-2019-09/dep_delay2_params.pkl')
    # # print(dep_delay2_params)
    # trip_time2_params = load('in/xtr/rt_20-2019-09/trip_time2_params.pkl')
    # sched_deps_out1 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound1.pkl')
    # sched_deps_out2 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound2.pkl')
    sim_arr_hws = []
    # sched_arr_out = load('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl')
    # observed_hw = load('in/xtr/rt_20-2019-09/arrival_headway_outbound.pkl')
    # sched_dep_in = load('in/xtr/rt_20-2019-09/departure_times_inbound.pkl')
    sim_dep_hws = []
    for i in range(4):
        # long pattern
        dep_delay1 = lognorm.rvs(dep_delay1_params[0], loc=dep_delay1_params[1], scale=dep_delay1_params[2], size=len(sched_deps_out1))
        trip_times1 = norm.rvs(loc=trip_time1_params[0], scale=trip_time1_params[1], size=len(sched_deps_out1))
        sim_arr1 = [i + j + k for i, j, k in zip(sched_deps_out1, dep_delay1, trip_times1)]
        # plt.hist(dep_delay1, bins=15, ec='black')
        # plt.show()
        # plt.close()
        # short pattern
        dep_delay2 = lognorm.rvs(dep_delay2_params[0], loc=dep_delay2_params[1], scale=dep_delay2_params[2], size=len(sched_deps_out2))
        trip_times2 = norm.rvs(loc=trip_time2_params[0], scale=trip_time2_params[1], size=len(sched_deps_out2))
        sim_arr2 = [i + j + k for i, j, k in zip(sched_deps_out2, dep_delay2, trip_times2)]
        # plt.hist(dep_delay2, bins=15, ec='black')
        # plt.show()
        # plt.close()
        # join and validate with scheduled arrivals and arrival headway
        sim_arr = sim_arr1 + sim_arr2
        sim_arr.sort()
        sim_arr_hw = [round(i - j) for i, j in zip(sim_arr[1:], sim_arr[:-1])]
        sim_arr_hws += sim_arr_hw
        sim_dep_hw = bus_availability(sched_arr_out, sched_dep_in, sim_arr, str(i))
        sim_dep_hws += sim_dep_hw

    sns.kdeplot(observed_hw, label='observed')
    sns.kdeplot(sim_arr_hws, label='simulated')
    plt.xlabel('outbound arrival headway (seconds)')
    plt.legend()
    plt.savefig('in/vis/validate_arrival_hw_out.png')
    plt.close()

    sns.kdeplot(dep_hw_in, label='observed')
    sns.kdeplot(sim_dep_hws, label='simulated')
    plt.xlabel('inbound departure headway (seconds)')
    plt.legend()
    plt.savefig('in/vis/validate_depart_hw_in.png')
    plt.close()
    return
