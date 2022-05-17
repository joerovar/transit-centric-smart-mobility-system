from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm


def get_interval(t, len_i_mins):
    # t is in seconds and len_i in minutes
    interval = int(t / (len_i_mins * 60))
    return interval


def remove_outliers(data):
    if data.any():
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.4 * iqr
        upper_bound = q3 + 1.4 * iqr
        data = data[(data >= lower_bound) & (data <= upper_bound)]
    return data


def get_route(path_stop_times, start_time_sec, end_time_sec, nr_intervals, start_interval, interval_length,
              dates, trip_choice, path_avl, delay_interval_length, delay_start_interval,
              tolerance_early_departure=1.5 * 60):
    stop_times_df = pd.read_csv(path_stop_times)
    avl_df = pd.read_csv(path_avl)

    df_for_stops = stop_times_df[stop_times_df['trip_id'] == trip_choice]
    df_for_stops = df_for_stops[df_for_stops['avl_arr_time'].astype(str).str[:10] == dates[0]]
    df_for_stops = df_for_stops.sort_values(by='stop_sequence')
    stops = df_for_stops['stop_id'].astype(str).tolist()

    df = stop_times_df[stop_times_df['stop_id'] == 386]
    df = df[df['stop_sequence'] == 1]
    df = df[df['schd_sec'] % 86400 <= end_time_sec]
    df = df[df['schd_sec'] % 86400 >= start_time_sec]
    df = df.sort_values(by='schd_sec')
    df = df.drop_duplicates(subset='trip_id')
    ordered_trip_ids = df['trip_id'].tolist()
    ordered_sched_dep = df['schd_sec'].tolist()
    ordered_block_ids = df['block_id'].tolist()
    ordered_schedule = []
    ordered_stops = []

    df_arrivals = stop_times_df[stop_times_df['trip_id'].isin(ordered_trip_ids)]
    df_arrivals = df_arrivals[df_arrivals['stop_sequence'] == 67]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals = df_arrivals.drop_duplicates(subset='trip_id')
    sched_arrivals = df_arrivals['schd_sec'].tolist()

    links = [str(s0) + '-' + str(s1) for s0, s1 in zip(stops[:-1], stops[1:])]
    link_times = {link: [[] for _ in range(nr_intervals)] for link in links}

    delay_nr_intervals = int(nr_intervals * interval_length/delay_interval_length)
    dep_delay_dist = [[] for _ in range(delay_nr_intervals)]
    dep_delay_ahead_dist = [[] for _ in range(delay_nr_intervals)]

    write_outbound_trajectories(path_avl, ordered_trip_ids)
    for t in ordered_trip_ids:
        temp = avl_df[avl_df['trip_id'] == t]
        temp = temp.sort_values(by='stop_sequence')

        temp_extract_schedule = temp.drop_duplicates(subset='stop_sequence')
        extract_schedule = temp_extract_schedule['schd_sec'].tolist()
        if len(extract_schedule) < 67:
            print(f'trip {t}')
        extract_stops = temp_extract_schedule['stop_id'].tolist()
        ordered_schedule.append(extract_schedule)
        ordered_stops.append([str(s) for s in extract_stops])

        for d in dates:
            date_specific = temp[temp['avl_arr_time'].astype(str).str[:10] == d]
            schd_sec = date_specific['schd_sec'].tolist()
            stop_id = date_specific['stop_id'].astype(str).tolist()
            avl_sec = date_specific['avl_arr_sec'].tolist()
            avl_dep_sec = date_specific['avl_dep_sec'].tolist()
            stop_sequence = date_specific['stop_sequence'].tolist()
            if len(avl_sec) > 1:
                if stop_sequence[0] == 1:
                    dep_delay = schd_sec[0] - (avl_dep_sec[0] % 86400)

                    delay_idx = get_interval(schd_sec[0], delay_interval_length) - delay_start_interval
                    dep_delay_dist[delay_idx].append(-1 * dep_delay)
                    if stop_sequence[1] == 2:
                        dep_delay_ahead = (avl_dep_sec[1] % 86400) - schd_sec[1]
                        dep_delay_ahead_dist[delay_idx].append(dep_delay_ahead)

                    if dep_delay > tolerance_early_departure:
                        schd_sec.pop(0)
                        stop_id.pop(0)
                        avl_sec.pop(0)
                        stop_sequence.pop(0)
                        avl_dep_sec.pop(0)
                for i in range(len(stop_id) - 1):
                    if stop_sequence[i] == stop_sequence[i + 1] - 1:
                        link = stop_id[i] + '-' + stop_id[i + 1]
                        if link in link_times:
                            nr_bin = get_interval(avl_sec[i] % 86400, interval_length) - start_interval
                            if 0 <= nr_bin < nr_intervals:
                                lt2 = avl_sec[i + 1] - avl_dep_sec[i]
                                if lt2 > 0:
                                    link_times[link][nr_bin].append(lt2)

    # clipping on 0, 300
    delay_min = -20
    delay_max = 250
    for i in range(len(dep_delay_dist)):
        arr = np.array(dep_delay_dist[i])
        arr = arr[(arr >= delay_min) & (arr<delay_max)]
        arr_ah = np.array(dep_delay_ahead_dist[i])
        arr_ah = arr_ah[(arr_ah>=delay_min) & (arr_ah<delay_max)]
        dep_delay_dist[i] = list(np.clip(arr, a_min=0, a_max=None))
        dep_delay_ahead_dist[i] = list(np.clip(arr_ah, a_min=0, a_max=None))
    fig, axs = plt.subplots(nrows=2, sharex='all')
    for i in range(2, 4):
        axs.flat[i - 2].hist([dep_delay_dist[i], dep_delay_ahead_dist[i]], label=['terminal', 'stop 1'])
    axs[0].set_title('outbound')
    plt.legend()
    plt.xlabel('dep delay (sec)')
    plt.savefig('in/vis/dep_delay_pk_out_clip.png')
    plt.close()

    mean_link_times = {link: [] for link in link_times}
    extreme_link_times = {link: [] for link in link_times}
    fit_params_link_t = {link: [] for link in link_times}
    for link in link_times:
        count = 0
        for interval_times in link_times[link]:
            if len(interval_times) > 1:
                interval_arr = np.array(interval_times)
                interval_arr = remove_outliers(interval_arr)
                fit_params = lognorm.fit(interval_arr, floc=0)
                fit_params_link_t[link].append(fit_params)

                mean = interval_arr.mean()
                cv = interval_arr.std() / mean
                if cv > 0.5:
                    print('suspicious link')
                    print(f'link {link} for interval {count}')
                    print(f'true cv {round(cv, 2)} with mean {round(mean, 2)} from {len(interval_times)} data')
                    new_dist = lognorm.rvs(*fit_params, size=30)
                    new_mean = new_dist.mean()
                    new_cv = np.std(new_dist) / new_mean
                    print(f'modeled cv {round(new_cv, 2)} with mean {round(new_mean, 2)}')

                extremes = (round(interval_arr.min()), round(interval_arr.max()))
                mean_link_times[link].append(mean)
                extreme_link_times[link].append(extremes)
            else:
                mean_link_times[link].append(np.nan)
                extreme_link_times[link].append(np.nan)
                fit_params_link_t[link].append(np.nan)
            count += 1

    # well known outlier link
    mean_link_times['3954-8613'][1] = mean_link_times['3954-8613'][3]
    mean_link_times['3954-8613'][2] = mean_link_times['3954-8613'][3]
    extreme_link_times['3954-8613'][1] = extreme_link_times['3954-8613'][3]
    extreme_link_times['3954-8613'][2] = extreme_link_times['3954-8613'][3]
    fit_params_link_t['3954-8613'][1] = fit_params_link_t['3954-8613'][3]
    fit_params_link_t['3954-8613'][2] = fit_params_link_t['3954-8613'][3]
    link_times_info = (mean_link_times, extreme_link_times, fit_params_link_t)

    trips_info = [(v, w, x, y, z) for v, w, x, y, z in zip(ordered_trip_ids, ordered_sched_dep, ordered_block_ids, ordered_schedule, ordered_stops)]
    return stops, trips_info, link_times_info, sched_arrivals, dep_delay_dist


def bi_proportional_fitting(od, target_ons, target_offs):
    balance_target_factor = np.sum(target_ons) / np.sum(target_offs)
    balanced_target_offs = target_offs * balance_target_factor
    for i in range(15):
        # balance rows
        actual_ons = np.nansum(od, axis=1)
        factor_ons = np.divide(target_ons, actual_ons, out=np.zeros_like(target_ons), where=actual_ons != 0)
        od = od * factor_ons[:, np.newaxis]

        # balance columns
        actual_offs = np.nansum(od, axis=0)
        factor_offs = np.divide(balanced_target_offs, actual_offs, out=np.zeros_like(target_offs),
                                where=actual_offs != 0)
        od = od * factor_offs

        # to check for tolerance we first assign 1.0 to totals of zero which cannot be changed by the method
        factor_ons[actual_ons == 0] = 1.0
        factor_offs[actual_offs == 0] = 1.0
    scaled_od_set = np.array(od)
    return scaled_od_set


def get_inbound_travel_time(path_stop_times, start_time, end_time, dates, nr_intervals,
                            start_interval, interval_length, path_avl, delay_interval_length,
                            delay_start_interval, tolerance_early_dep=0.5 * 60):
    # dep_delay_record_short = []
    # dep_delay_record_long = []
    trip_time_record_long = []
    trip_time_record_short = []

    stop_times_df = pd.read_csv(path_stop_times)
    avl_df = pd.read_csv(path_avl)

    df1 = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df1 = df1[df1['stop_id'] == 8613]
    df1 = df1[df1['schd_sec'] <= end_time]
    df1 = df1[df1['schd_sec'] >= start_time]
    df1 = df1.drop_duplicates(subset='trip_id')
    df1 = df1.sort_values(by='schd_sec')
    ordered_trip_ids1 = df1['trip_id'].tolist()
    ordered_deps1 = df1['schd_sec'].tolist()
    ordered_block_ids1 = df1['block_id'].tolist()

    add_sched_dep_time = (datetime.strptime('7:33:30', '%H:%M:%S') - datetime(1900, 1, 1)).total_seconds()
    add_trip_id = 911266020
    add_block_id_df = stop_times_df[stop_times_df['trip_id'] == add_trip_id]
    add_block_id = int(add_block_id_df['block_id'].mean())
    for i in range(1, len(ordered_trip_ids1) - 1):
        if ordered_deps1[i - 1] < add_sched_dep_time < ordered_deps1[i]:
            idx_insert = i - 1
            break
    ordered_trip_ids1.insert(idx_insert, add_trip_id)
    ordered_deps1.insert(idx_insert, add_sched_dep_time)
    ordered_block_ids1.insert(idx_insert, add_block_id)

    df2 = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df2 = df2[df2['stop_id'] == 15136]
    df2 = df2[df2['schd_sec'] <= end_time]
    df2 = df2[df2['schd_sec'] >= start_time]
    df2 = df2.sort_values(by='schd_sec')
    df2 = df2.drop_duplicates(subset='trip_id')
    ordered_trip_ids2 = df2['trip_id'].tolist()
    ordered_deps2 = df2['schd_sec'].tolist()
    ordered_block_ids2 = df2['block_id'].tolist()

    all_trip_ids = ordered_trip_ids1 + ordered_trip_ids2
    df_arrivals = stop_times_df[stop_times_df['trip_id'].isin(all_trip_ids)]
    df_arrivals = df_arrivals[df_arrivals['stop_sequence'].isin([23, 63])]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals = df_arrivals.drop_duplicates(subset='trip_id')
    df_arrivals = df_arrivals[df_arrivals['schd_sec'] <= end_time]
    ordered_arriving_trip_ids = df_arrivals['trip_id'].tolist()
    sched_arrivals = df_arrivals['schd_sec'].tolist()

    trip_times1 = [[] for _ in range(nr_intervals)]

    delay_nr_intervals = int(nr_intervals * interval_length/delay_interval_length)
    dep_delay1 = [[] for _ in range(delay_nr_intervals)]
    dep_delay_1_ahead = [[] for _ in range(delay_nr_intervals)]
    trip_t1_empirical = [[] for _ in range(nr_intervals)]
    # deadhead_times = [[] for _ in range(nr_intervals)]
    # stop_seq_terminal2 = 41
    # dep_delay1 = []
    for t in ordered_trip_ids1:
        temp_df = avl_df[avl_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 63:
                    arrival_sec = df['avl_arr_sec'].tolist()
                    dep_sec = df['avl_dep_sec'].tolist()
                    schd_sec = df['schd_sec'].tolist()
                    dep_delay = schd_sec[0] - (dep_sec[0] % 86400)
                    # dep_delay1.append(-dep_delay)
                    if (schd_sec[0] > start_time + 3600) and (schd_sec[0] < end_time - 6400):
                        trip_time_record_long.append(arrival_sec[-1] - dep_sec[0])

                    delay_idx = get_interval(schd_sec[0], delay_interval_length) - delay_start_interval
                    trip_t_idx = get_interval(schd_sec[0], interval_length) - start_interval
                    dep_delay1[delay_idx].append(-1 * dep_delay)
                    trip_t1_empirical[trip_t_idx].append(arrival_sec[-1] - dep_sec[0])
                    if stop_seq[1] == 2:
                        dep_delay_ahead = schd_sec[1] - (dep_sec[1] % 86400)
                        dep_delay_1_ahead[delay_idx].append(-1 * dep_delay_ahead)

                    if dep_delay < tolerance_early_dep:
                        idx = get_interval(schd_sec[0], interval_length) - start_interval
                        trip_times1[idx].append(arrival_sec[-1] - dep_sec[0])

                # if stop_seq[0] == 1 and stop_seq_terminal2 in stop_seq:
                #     deadhead_df = df[df['stop_sequence'] <= stop_seq_terminal2]
                #     deadhead_dep_t = deadhead_df['avl_dep_sec'].tolist()[0]
                #     schd_dep_t = deadhead_df['schd_sec'].tolist()[0]
                #     deadhead_arr_t = deadhead_df['avl_arr_sec'].tolist()[-1]
                #     if schd_dep_t - (deadhead_dep_t % 86400) < tolerance_early_dep:
                #         # we take out the departure terminal because of faulty dwell time measurements
                #         deadhead_df = deadhead_df[deadhead_df['stop_sequence'] > 1]
                #         dwell_times = deadhead_df['dwell_time'].sum()
                #         deadhead_time = (deadhead_arr_t - deadhead_dep_t) - dwell_times
                #         idx = get_interval(schd_dep_t, interval_length) - start_interval
                #         deadhead_times[idx].append(deadhead_time)
    trip_times2 = [[] for _ in range(nr_intervals)]
    dep_delay2 = [[] for _ in range(delay_nr_intervals)]
    dep_delay_2_ahead = [[] for _ in range(delay_nr_intervals)]
    trip_t2_empirical = [[] for _ in range(nr_intervals)]
    # dep_delay2 = []
    for t in ordered_trip_ids2:
        temp_df = avl_df[avl_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            if stop_seq:
                if stop_seq[0] == 1 and stop_seq[-1] == 23:
                    arrival_sec = df['avl_arr_sec'].tolist()
                    dep_sec = df['avl_dep_sec'].tolist()
                    schd_sec = df['schd_sec'].tolist()
                    dep_delay = schd_sec[0] - (dep_sec[0] % 86400)
                    if (schd_sec[0] > start_time + 3600) and (schd_sec[0] < end_time - 6400):
                        trip_time_record_short.append(arrival_sec[-1] - dep_sec[0])

                    delay_idx = get_interval(schd_sec[0], delay_interval_length) - delay_start_interval
                    trip_t_idx = get_interval(schd_sec[0], delay_interval_length) - start_interval
                    dep_delay2[delay_idx].append(-1 * dep_delay)
                    if dep_delay <tolerance_early_dep:
                        trip_t2_empirical[trip_t_idx].append(arrival_sec[-1] - dep_sec[0])
                    if stop_seq[1] == 2:
                        dep_delay_ahead = schd_sec[1] - (dep_sec[1] % 86400)
                        dep_delay_2_ahead[delay_idx].append(-1 * dep_delay_ahead)

                    if dep_delay < tolerance_early_dep:
                        idx = get_interval(schd_sec[0], interval_length) - start_interval
                        trip_times2[idx].append(arrival_sec[-1] - dep_sec[0])
    # clipping 1 (long) on 0, 300
    delay1_min = -20
    delay1_max = 300
    for i in range(len(dep_delay1)):
        arr = np.array(dep_delay1[i])
        arr = arr[(arr>=delay1_min) & (arr<delay1_max)]
        arr_ah = np.array(dep_delay_1_ahead[i])
        arr_ah = arr_ah[(arr_ah>delay1_min) & (arr_ah<delay1_max)]
        dep_delay1[i] = list(np.clip(arr, a_min=0, a_max=None))
        dep_delay_1_ahead[i] = list(np.clip(arr_ah, a_min=0, a_max=None))

    # clipping 2 (short) on 0, 200
    delay2_min = -20
    delay2_max = 200
    for i in range(len(dep_delay2)):
        arr = np.array(dep_delay2[i])
        arr = arr[(arr>=delay2_min) & (arr<delay2_max)]
        arr_ah = np.array(dep_delay_2_ahead[i])
        arr_ah = arr_ah[(arr_ah>delay2_min) & (arr_ah<delay2_max)]

        dep_delay2[i] = list(np.clip(arr, a_min=0,a_max=None))
        dep_delay_2_ahead[i] = list(np.clip(arr_ah, a_min=0, a_max=None))

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
    for t_idx in range(2, 4):
        dep_delay_arr = np.array(dep_delay1[t_idx])
        dep_delay_ahead_arr = np.array(dep_delay_1_ahead[t_idx])
        dep_delay2_arr = np.array(dep_delay2[t_idx])
        dep_delay2_ahead_arr = np.array(dep_delay_2_ahead[t_idx])
        axs[t_idx - 2, 0].hist([dep_delay_arr, dep_delay_ahead_arr],
                               label=['terminal', 'stop 1'])
        axs[t_idx - 2, 1].hist([dep_delay2_arr, dep_delay2_ahead_arr],
                               label=['terminal', 'stop 1'])
    axs[-1, 0].set_xlabel('dep delay (sec)')
    axs[-1, 1].set_xlabel('dep delay (sec)')
    axs[0, 0].set_title('long')
    axs[0, 1].set_title('short')
    plt.legend()
    plt.tight_layout()
    plt.savefig('in/vis/dep_delay_pk_in_clipped.png')
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
    for t_idx in range(2, 4):
        arr1 = remove_outliers(np.array(trip_t1_empirical[t_idx]))
        arr2 = remove_outliers(np.array(trip_t2_empirical[t_idx]))
        trip_t1_empirical[t_idx] = list(arr1)
        trip_t2_empirical[t_idx] = list(arr2)
        axs[t_idx - 2, 0].hist(arr1/60, ec='black')
        if arr2.size:
            axs[t_idx - 2, 1].hist(arr2/60, ec='black')
        # dep_delay_arr = np.array(dep_delay1[t_idx])
        # dep_delay_ahead_arr = np.array(dep_delay_1_ahead[t_idx])
        # dep_delay2_arr = np.array(dep_delay2[t_idx])
        # dep_delay2_ahead_arr = np.array(dep_delay_2_ahead[t_idx])
        # axs[t_idx - 2, 0].hist([dep_delay_arr, dep_delay_ahead_arr],
        #                        label=['terminal', 'stop 1'])
        # axs[t_idx - 2, 1].hist([dep_delay2_arr, dep_delay2_ahead_arr],
        #                        label=['terminal', 'stop 1'])
    axs[-1, 0].set_xlabel('trip time (min)')
    axs[-1, 1].set_xlabel('trip time (min)')
    axs[0, 0].set_title('long')
    axs[0, 1].set_title('short')
    plt.tight_layout()
    plt.savefig('in/vis/trip_t_in_hist.png')
    plt.close()

    arrival_headway = []
    df = avl_df[avl_df['trip_id'].isin(all_trip_ids)]
    df = df.sort_values(by=['stop_sequence', 'schd_sec'])
    df.to_csv('in/vis/trajectories_inbound.csv', index=False)
    df = avl_df[avl_df['trip_id'].isin(ordered_arriving_trip_ids)]
    df = df.sort_values(by=['stop_sequence', 'schd_sec'])
    df_arrivals = df[df['stop_id'] == 386]
    df_arrivals = df_arrivals.sort_values(by='schd_sec')
    df_arrivals.to_csv('in/vis/arrivals_inbound.csv', index=False)
    for d in dates:
        temp_df = df[df['avl_arr_time'].astype(str).str[:10] == d]
        temp_df = temp_df[temp_df['stop_id'] == 386]
        temp_df = temp_df.sort_values(by='schd_sec')
        avl_sec = temp_df['avl_arr_sec'].tolist()
        if avl_sec:
            avl_sec.sort()
            temp_arr_hw = [i - j for i, j in zip(avl_sec[1:], avl_sec[:-1])]
            arrival_headway += temp_arr_hw
    trip_times1_params = []
    trip_times2_params = []
    # deadhead_times_params = []
    for i in range(nr_intervals):
        trip_times1[i] = remove_outliers(np.array(trip_times1[i])).tolist()
        trip_times2[i] = remove_outliers(np.array(trip_times2[i])).tolist()
        # deadhead_times[i] = remove_outliers(np.array(deadhead_times[i])).tolist()

        lognorm_params1 = lognorm.fit(trip_times1[i], floc=0)
        lognorm_params2 = lognorm.fit(trip_times2[i], floc=0)

        trip_times1_params.append(lognorm_params1)
        trip_times2_params.append(lognorm_params2)
        # deadhead_times_params.append(norm.fit(deadhead_times[i]))
    trips1_info = [(x, y, z) for x, y, z in zip(ordered_trip_ids1, ordered_deps1, ordered_block_ids1)]
    trips2_info = [(x, y, z) for x, y, z in zip(ordered_trip_ids2, ordered_deps2, ordered_block_ids2)]

    all_trip_times = [trip_time_record_long, trip_time_record_short]

    # fill in the nans for extreme cases
    # trip_times1_params[1] = trip_times1_params[2]
    # trip_times2_params[-4] = trip_times2_params[-5]
    # trip_times1_params[-1] = trip_times1_params[-2]
    return trips1_info, trips2_info, sched_arrivals, trip_times1_params, trip_times2_params, dep_delay1, dep_delay2, trip_t1_empirical, trip_t2_empirical


def analyze_inbound(path_avl, start_time, end_time, delay_interval_length):
    avl_df = pd.read_csv(path_avl)

    end_terminal_id = 386
    terminal_seq_long = 63
    terminal_seq_short = 23

    arr_delays_long = []
    arr_delays_short = []
    # arrivals
    arr_long_df = avl_df[avl_df['stop_id'] == end_terminal_id]
    arr_long_df = arr_long_df[arr_long_df['stop_sequence'] == terminal_seq_long]
    arr_long_df['avl_arr_sec'] = arr_long_df['avl_arr_sec'] % 86400

    arr_short_df = avl_df[avl_df['stop_id'] == end_terminal_id]
    arr_short_df = arr_short_df[arr_short_df['stop_sequence'] == terminal_seq_short]
    arr_short_df['avl_arr_sec'] = arr_short_df['avl_arr_sec'] % 86400

    start_terminal_id_long = 8613
    start_terminal_id_short = 15136

    dep_delays_long = []
    dep_delays_short = []
    # departures
    dep_long_df = avl_df[avl_df['stop_id'] == start_terminal_id_long]
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == 1]
    dep_long_df['avl_dep_sec'] = dep_long_df['avl_dep_sec'] % 86400

    dep_short_df = avl_df[avl_df['stop_id'] == start_terminal_id_short]
    dep_short_df = dep_short_df[dep_short_df['stop_sequence'] == 1]
    dep_short_df['avl_dep_sec'] = dep_short_df['avl_dep_sec'] % 86400

    interval0 = get_interval(start_time, delay_interval_length)
    interval1 = get_interval(end_time, delay_interval_length)

    for interval in range(interval0, interval1):
        # arrivals
        temp_df = arr_long_df[arr_long_df['schd_sec']>=interval*delay_interval_length*60]
        temp_df = temp_df[temp_df['schd_sec']<=(interval+1)*delay_interval_length*60]
        temp_df['delay'] = temp_df['avl_arr_sec'] - temp_df['schd_sec']
        d = remove_outliers(temp_df['delay'].to_numpy())
        arr_delays_long.append(d.tolist())

        temp_df = arr_short_df[arr_short_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df['avl_arr_sec'] - temp_df['schd_sec']
        d = remove_outliers(temp_df['delay'].to_numpy())
        arr_delays_short.append(d.tolist())

        # departures
        temp_df = dep_long_df[dep_long_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df['avl_dep_sec'] - temp_df['schd_sec']
        d = remove_outliers(temp_df['delay'].to_numpy())
        dep_delays_long.append(d.tolist())

        temp_df = dep_short_df[dep_short_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df['avl_dep_sec'] - temp_df['schd_sec']
        d = remove_outliers(temp_df['delay'].to_numpy())
        dep_delays_short.append(d.tolist())

    # short
    fig, ax = plt.subplots(nrows=3, ncols=2)
    for i in range(len(arr_delays_short)):
        ax.flat[i].hist([dep_delays_short[i], arr_delays_short[i]])
    plt.show()
    plt.close()

    fig, ax = plt.subplots(nrows=3, ncols=2)
    for i in range(len(arr_delays_long)):
        ax.flat[i].hist([dep_delays_long[i], arr_delays_long[i]])
    plt.show()
    plt.close()
    return


def get_trip_times(path_avl, focus_trips, dates, stops):
    # TRIP TIMES ARE USED FOR CALIBRATION THEREFORE THEY ONLY USE THE FOCUS TRIPS FOR RESULTS ANALYSIS
    trip_times = []
    stop_times_df = pd.read_csv(path_avl)
    # extra_stop_times_df = pd.read_csv(path_extra_stop_times)
    arrival_times = {}
    for t in focus_trips:
        temp_df = stop_times_df[stop_times_df['trip_id'] == t]
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d]
            df = df.sort_values(by='stop_sequence')
            stop_seq = df['stop_sequence'].tolist()
            arrival_sec = df['avl_arr_sec'].tolist()
            dep_sec = df['avl_dep_sec'].tolist()
            if stop_seq:
                for s_idx in range(len(stop_seq)):
                    if stop_seq[s_idx] != 1:
                        ky = str(t) + str(d) + str(stops[stop_seq[s_idx] - 1])
                        vl = arrival_sec[s_idx]
                        arrival_times[ky] = vl
                if 2 in stop_seq and 66 in stop_seq:
                    dep_idx = stop_seq.index(2)
                    arr_idx = stop_seq.index(66)
                    tt = arrival_sec[arr_idx] - dep_sec[dep_idx]
                    if tt > 57 * 60:
                        trip_times.append(tt)
    hws = [[] for _ in stops]
    for d in dates:
        date_df = stop_times_df[stop_times_df['avl_arr_time'].astype(str).str[:10] == d]
        date_df = date_df[date_df['trip_id'].isin(focus_trips)]
        for j in range(len(stops)):
            df = date_df[date_df['stop_id'] == int(stops[j])]
            df = df.sort_values(by='avl_arr_sec')
            arr_sec = df['avl_arr_sec'].tolist()
            if len(arr_sec) > 1:
                for i in range(1, len(arr_sec)):
                    hws[j].append(arr_sec[i] - arr_sec[i-1])
    cv_hws = []
    for i in range(len(hws)):
        hws[i] = remove_outliers(np.array(hws[i])).tolist()
        print(hws[i])
        cv_hws.append(np.std(hws[i]) / np.mean(hws[i]))
    # PROCESS HEADWAY
    hw_in_all = {s: [] for s in stops[1:]}
    hw_in_cv = []
    for d in dates:
        for s in stops[1:]:
            for n in range(1, len(focus_trips)):
                ky0 = str(focus_trips[n - 1]) + str(d) + str(s)
                ky1 = str(focus_trips[n]) + str(d) + str(s)
                if ky0 in arrival_times and ky1 in arrival_times:
                    hwt = arrival_times[ky1] - arrival_times[ky0]
                    if hwt > 0:
                        hw_in_all[s].append(hwt)
    for s in stops[1:]:
        if hw_in_all[s]:
            hws = remove_outliers(np.array(hw_in_all[s])).tolist()
            # print(hws)
            hw_in_cv.append(np.std(hws) / np.mean(hws))
    # THIS WE WILL USE TO VALIDATE THE INBOUND DIRECTION MODELING HENCE WE TAKE ALL DEPARTURES IN THE PERIOD OF STUDY
    hw = []
    df = stop_times_df[stop_times_df['stop_id'] == int(stops[30])]
    df = df[df['stop_sequence'] == 31]
    df = df[df['trip_id'].isin(focus_trips)]
    df = df.sort_values(by='schd_sec')
    for d in dates:
        temp_df = df[df['avl_arr_time'].astype(str).str[:10] == d]
        if not temp_df.empty:
            avl_dep_sec = temp_df['avl_dep_sec'].to_list()
            trip_ids = temp_df['trip_id'].tolist()
            for i in range(1, len(trip_ids)):
                idx = focus_trips.index(trip_ids[i])
                if trip_ids[i - 1] == focus_trips[idx - 1]:
                    h = avl_dep_sec[i] - avl_dep_sec[i - 1]
                    if h < 10:
                        print('base')
                        print(f'trips {trip_ids[i]} and {trip_ids[i - 1]}')
                        print(f'times {avl_dep_sec[i]} and {avl_dep_sec[i - 1]}')
                    hw.append(h)
    hw_arr = np.array(hw)
    hw_arr = hw_arr[hw_arr >= 15]
    hw = remove_outliers(hw_arr).tolist()


    return trip_times, hw, hw_in_cv, cv_hws


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


def write_outbound_trajectories(stop_times_path, ordered_trips):
    stop_times_df = pd.read_csv(stop_times_path)
    stop_times_df = stop_times_df[stop_times_df['trip_id'].isin(ordered_trips)]
    stop_times_df = stop_times_df.sort_values(by=['stop_sequence', 'schd_sec'])
    stop_times_df.to_csv('in/vis/trajectories_outbound.csv', index=False)
    return


def get_load_profile(stop_times_path, focus_trips, stops):
    lp = []
    ons_set = []
    offs_set = []
    stop_times_df = pd.read_csv(stop_times_path)
    stop_times_df = stop_times_df[stop_times_df['trip_id'].isin(focus_trips)]
    for s in stops:
        df = stop_times_df[stop_times_df['stop_id'] == int(s)]
        ons_df = df['ron'] + df['fon']
        ons = remove_outliers(ons_df.to_numpy())
        ons_set.append(ons.mean())
        offs_df = df['roff'] + df['foff']
        offs = remove_outliers(offs_df.to_numpy())
        offs_set.append(offs.mean())
        lp_dataset = remove_outliers(df['passenger_load'].to_numpy())
        lp.append(lp_dataset.mean())
    return lp, ons_set, offs_set


def get_scheduled_bus_availability(path_stop_times, dates, start_time, end_time):
    stop_times_df = pd.read_csv(path_stop_times)
    date = dates[2]
    stop_times_df = stop_times_df[stop_times_df['avl_arr_time'].astype(str).str[:10] == date]

    df_inbound = stop_times_df[stop_times_df['stop_sequence'].isin([23, 63])]
    df_inbound = df_inbound[df_inbound['stop_id'] == 386]
    df_inbound = df_inbound[df_inbound['schd_sec'] >= start_time]
    df_inbound = df_inbound[df_inbound['schd_sec'] <= end_time]
    df_inbound = df_inbound.sort_values(by='avl_arr_sec')
    actual_arrivals = df_inbound['avl_arr_sec'] % 86400
    actual_arrivals = actual_arrivals.tolist()
    df_inbound = df_inbound.sort_values(by='schd_sec')
    scheduled_arrivals = df_inbound['schd_sec'].tolist()

    df_outbound = stop_times_df[stop_times_df['stop_sequence'] == 1]
    df_outbound = df_outbound[df_outbound['stop_id'] == 386]
    df_outbound = df_outbound[df_outbound['schd_sec'] >= start_time]
    df_outbound = df_outbound[df_outbound['schd_sec'] <= end_time]
    df_outbound = df_outbound.sort_values(by='avl_dep_sec')
    actual_departures = df_outbound['avl_dep_sec'] % 86400
    actual_departures = actual_departures.tolist()
    df_outbound = df_outbound.sort_values(by='schd_sec')
    scheduled_departures = df_outbound['schd_sec'].tolist()

    plt.plot(scheduled_departures, [i for i in range(1, len(scheduled_departures) + 1)], '--',
             label='scheduled_departures', color='red')
    plt.plot(scheduled_arrivals, [i for i in range(1, len(scheduled_arrivals) + 1)], '--', label='scheduled arrivals',
             color='green')
    plt.plot(actual_arrivals, [i for i in range(1, len(actual_arrivals) + 1)], label='avl arrivals')
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
    plt.plot(sched_arr, np.arange(1, len(sched_arr) + 1), '--', label='scheduled arrivals', color='green', alpha=0.5)
    plt.plot(sched_dep, np.arange(1, len(sched_dep) + 1), '--', label='scheduled departures', color='red', alpha=0.5)
    plt.plot(actual_arr, np.arange(1, len(actual_arr) + 1), label='actual arrivals', color='green')
    plt.plot(actual_dep, np.arange(1, len(actual_dep) + 1), label='actual departures', color='red')
    plt.legend()
    plt.savefig('in/vis/sample_bus_availability' + iden + '.png')

    plt.close()
    actual_dep_hw = [j - k for j, k in zip(actual_dep[1:], actual_dep[:-1])]
    return actual_dep_hw


def extract_apc_counts(nr_intervals, odt_ordered_stops, path_stop_times, interval_len_min, dates):
    arr_rates = np.zeros(shape=(nr_intervals, len(odt_ordered_stops)))
    drop_rates = np.zeros(shape=(nr_intervals, len(odt_ordered_stops)))
    stop_t_df = pd.read_csv(path_stop_times)
    for stop_idx in range(len(odt_ordered_stops)):
        print(f'stop {stop_idx + 1}')
        temp_df = stop_t_df[stop_t_df['stop_id'] == int(odt_ordered_stops[stop_idx])]
        for interval_idx in range(48):
            t_edge0 = interval_idx * interval_len_min * 60
            t_edge1 = (interval_idx + 1) * interval_len_min * 60
            pax_df = temp_df[temp_df['avl_dep_sec'] % 86400 <= t_edge1]
            pax_df = pax_df[pax_df['avl_dep_sec'] % 86400 >= t_edge0]
            ons_rate_by_date = np.zeros(len(dates))
            ons_rate_by_date[:] = np.nan
            for k in range(len(dates)):
                day_df = pax_df[pax_df['avl_arr_time'].astype(str).str[:10] == dates[k]]
                if not day_df.empty:
                    ons_rate_by_date[k] = (day_df['ron'].sum() + day_df['fon'].sum()) * 60 / interval_len_min
            all_nan = True not in np.isfinite(ons_rate_by_date)
            if not all_nan:
                arr_rates[interval_idx, stop_idx] = np.nanmean(ons_rate_by_date)
            offs_rate_by_date = np.zeros(len(dates))
            offs_rate_by_date[:] = np.nan
            for k in range(len(dates)):
                day_df = pax_df[pax_df['avl_arr_time'].astype(str).str[:10] == dates[k]]
                if not day_df.empty:
                    offs_rate_by_date[k] = (day_df['roff'].sum() + day_df['foff'].sum()) * 60 / interval_len_min
            all_nan = True not in np.isfinite(offs_rate_by_date)
            if not all_nan:
                drop_rates[interval_idx, stop_idx] = np.nanmean(offs_rate_by_date)

    return arr_rates, drop_rates
