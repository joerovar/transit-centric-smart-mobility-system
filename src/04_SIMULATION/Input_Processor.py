from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pickle
from File_Paths import path_route_stops, path_apc_counts, path_avl


def extract_demand(odt_interval_len_min, dates):
    stops_outbound = load(path_route_stops)
    odt_stops = np.load('in/xtr/rt_20_odt_stops.npy')
    # comes from project with dingyi data
    odt_pred = np.load('in/xtr/rt_20_odt_rates_30.npy')
    # comes from project with dingyi data

    nr_intervals = 24 / (odt_interval_len_min / 60)
    apc_on_rates, apc_off_rates = extract_apc_counts(int(nr_intervals), odt_stops, odt_interval_len_min,
                                                     dates)
    stops_lst = list(odt_stops)

    # DISCOVERED IN DINGYI'S OD MATRIX TIME SHIFT
    shifted_odt = np.concatenate((odt_pred[-6:], odt_pred[:-6]), axis=0)
    scaled_odt = np.concatenate((odt_pred[-6:], odt_pred[:-6]), axis=0)

    for i in range(shifted_odt.shape[0]):
        print(f'interval {i}')
        scaled_odt[i] = bi_proportional_fitting(shifted_odt[i], apc_on_rates[i], apc_off_rates[i])

    np.save('in/xtr/rt_20_odt_rates_30_scaled.npy', scaled_odt)

    # if wanted for comparison
    idx_stops_out = [stops_lst.index(int(s)) for s in stops_outbound]
    out_on_counts = apc_on_rates[:, idx_stops_out]
    out_on_tot_count = np.nansum(out_on_counts, axis=-1)

    arr_rates_shifted = np.nansum(shifted_odt, axis=-1)
    out_arr_rates_shifted = arr_rates_shifted[:, idx_stops_out]
    out_arr_tot_shifted = np.sum(out_arr_rates_shifted, axis=-1)

    scaled_arr_rates = np.sum(scaled_odt, axis=-1)
    scaled_out_arr_rates = scaled_arr_rates[:, idx_stops_out]
    scaled_out_tot = np.sum(scaled_out_arr_rates, axis=-1)

    x = np.arange(out_on_tot_count.shape[0])
    plt.plot(x, scaled_out_tot, label='odt scaled')
    plt.plot(x, out_arr_tot_shifted, label='odt')
    plt.plot(x, out_on_tot_count, label='apc')
    plt.xticks(np.arange(0, out_on_tot_count.shape[0], 2), np.arange(int(out_on_tot_count.shape[0] / 2)))
    plt.xlabel('hour of day')
    plt.ylabel('arrival rate (1/h)')
    plt.yticks(np.arange(0, 1200, 200))
    plt.legend()
    # plt.show()
    plt.close()
    return


def load(pathname):
    with open(pathname, 'rb') as tf:
        var = pickle.load(tf)
    return var


def save(pathname, par):
    with open(pathname, 'wb') as tf:
        pickle.dump(par, tf)
    return


def get_interval(t, len_i_mins):
    # t is in seconds and len_i in minutes
    interval = int(t / (len_i_mins * 60))
    return interval


def remove_outliers(data, factor=1.4):
    # 1.5 is preferred
    if data.any():
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        data = data[(data >= lower_bound) & (data <= upper_bound)]
    return data


def extract_outbound_params(start_time_sec, end_time_sec, nr_intervals, start_interval,
                            interval_length, dates, trip_choice, delay_interval_length, delay_start_interval,
                            tolerance_early_departure=1.5 * 60):
    # stop_times_df = pd.read_csv(path_stop_times)
    avl_df = pd.read_csv(path_avl)

    df_for_stops = avl_df[avl_df['trip_id'] == trip_choice].copy()
    df_for_stops = df_for_stops[df_for_stops['avl_arr_time'].astype(str).str[:10] == dates[0]]
    df_for_stops = df_for_stops.sort_values(by='stop_sequence')
    stops = df_for_stops['stop_id'].astype(str).tolist()
    print(stops)

    df = avl_df[avl_df['stop_id'] == 386].copy()
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

    links = [str(s0) + '-' + str(s1) for s0, s1 in zip(stops[:-1], stops[1:])]
    link_times = {link: [[] for _ in range(nr_intervals)] for link in links}

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)
    dep_delay_dist = [[] for _ in range(delay_nr_intervals)]
    dep_delay_ahead_dist = [[] for _ in range(delay_nr_intervals)]

    write_outbound_trajectories(path_avl, ordered_trip_ids)
    for t in ordered_trip_ids:
        temp = avl_df[avl_df['trip_id'] == t].copy()
        temp = temp.sort_values(by='stop_sequence')

        temp_extract_schedule = temp.drop_duplicates(subset='stop_sequence')
        extract_schedule = temp_extract_schedule['schd_sec'].tolist()
        if len(extract_schedule) < 67:
            print(f'trip {t}')
        extract_stops = temp_extract_schedule['stop_id'].tolist()
        ordered_schedule.append(extract_schedule)
        ordered_stops.append([str(s) for s in extract_stops])

        for d in dates:
            date_specific = temp[temp['avl_arr_time'].astype(str).str[:10] == d].copy()
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
        arr = arr[(arr >= delay_min) & (arr < delay_max)]
        arr_ah = np.array(dep_delay_ahead_dist[i])
        arr_ah = arr_ah[(arr_ah >= delay_min) & (arr_ah < delay_max)]
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
                    # print('suspicious link')
                    # print(f'link {link} for interval {count}')
                    # print(f'true cv {round(cv, 2)} with mean {round(mean, 2)} from {len(interval_times)} data')
                    new_dist = lognorm.rvs(*fit_params, size=30)
                    new_mean = new_dist.mean()
                    new_cv = np.std(new_dist) / new_mean
                    # print(f'modeled cv {round(new_cv, 2)} with mean {round(new_mean, 2)}')

                extremes = (round(interval_arr.min()), round(interval_arr.max()))
                mean_link_times[link].append(mean)
                extreme_link_times[link].append(extremes)
            else:
                mean_link_times[link].append(np.nan)
                extreme_link_times[link].append(np.nan)
                fit_params_link_t[link].append(np.nan)
            count += 1

    # well known outlier link
    link_times_info = (mean_link_times, extreme_link_times, fit_params_link_t)

    trips_info = [(v, w, x, y, z) for v, w, x, y, z in
                  zip(ordered_trip_ids, ordered_sched_dep, ordered_block_ids, ordered_schedule, ordered_stops)]

    stop_df = pd.read_csv('in/raw/gtfs/stops.txt')
    stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in stops])]

    stop_seq_dict = {'stop_id': [int(s) for s in stops], 'stop_seq': [i for i in range(1, len(stops) + 1)]}
    stop_seq_df = pd.DataFrame(stop_seq_dict)
    stop_df = pd.merge(stop_df, stop_seq_df, on='stop_id')
    stop_df = stop_df.sort_values(by='stop_seq')
    stop_df = stop_df[['stop_seq', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
    stop_df.to_csv('in/raw/rt20_in_stops.txt', index=False)

    save('in/xtr/route_stops.pkl', stops)
    save('in/xtr/link_times_info.pkl', link_times_info)
    save('in/xtr/trips_outbound_info.pkl', trips_info)
    save('in/xtr/dep_delay_dist_out.pkl', dep_delay_dist)
    return


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


def extract_inbound_params(start_time, end_time, dates, nr_intervals,
                           start_interval, interval_length, delay_interval_length,
                           delay_start_interval, tolerance_early_dep=0.5 * 60):
    trip_time_record_long = []
    trip_time_record_short = []

    # stop_times_df = pd.read_csv(path_stop_times)
    avl_df = pd.read_csv(path_avl)

    df1 = avl_df[avl_df['stop_sequence'] == 1].copy()
    df1 = df1[df1['stop_id'] == 8613]
    df1 = df1[df1['schd_sec'] <= end_time]
    df1 = df1[df1['schd_sec'] >= start_time]
    df1 = df1.drop_duplicates(subset='trip_id')
    df1 = df1.sort_values(by='schd_sec')
    ordered_trip_ids1 = df1['trip_id'].tolist()
    ordered_deps1 = df1['schd_sec'].tolist()
    ordered_block_ids1 = df1['block_id'].tolist()
    ordered_schedules1 = []
    ordered_stops1 = []

    # add_sched_dep_time = (datetime.strptime('7:33:30', '%H:%M:%S') - datetime(1900, 1, 1)).total_seconds()
    # add_trip_id = 911266020
    # add_block_id_df = avl_df[avl_df['trip_id'] == add_trip_id].copy()
    # add_block_id = int(add_block_id_df['block_id'].mean())
    # for i in range(1, len(ordered_trip_ids1) - 1):
    #     if ordered_deps1[i - 1] < add_sched_dep_time < ordered_deps1[i]:
    #         idx_insert = i - 1
    #         break
    # ordered_trip_ids1.insert(idx_insert, add_trip_id)
    # ordered_deps1.insert(idx_insert, add_sched_dep_time)
    # ordered_block_ids1.insert(idx_insert, add_block_id)

    df2 = avl_df[avl_df['stop_sequence'] == 1].copy()
    df2 = df2[df2['stop_id'] == 15136]
    df2 = df2[df2['schd_sec'] <= end_time]
    df2 = df2[df2['schd_sec'] >= start_time]
    df2 = df2.sort_values(by='schd_sec')
    df2 = df2.drop_duplicates(subset='trip_id')
    ordered_trip_ids2 = df2['trip_id'].tolist()
    ordered_deps2 = df2['schd_sec'].tolist()
    ordered_block_ids2 = df2['block_id'].tolist()
    ordered_schedules2 = []
    ordered_stops2 = []

    trip_times1 = [[] for _ in range(nr_intervals)]

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)
    dep_delay1 = [[] for _ in range(delay_nr_intervals)]
    dep_delay_1_ahead = [[] for _ in range(delay_nr_intervals)]
    trip_t1_empirical = [[] for _ in range(nr_intervals)]
    for t in ordered_trip_ids1:
        temp_df = avl_df[avl_df['trip_id'] == t].copy()
        schedule_df = temp_df.drop_duplicates(subset='schd_sec')
        schedule_df = schedule_df.sort_values(by='schd_sec')
        ordered_schedules1.append(schedule_df['schd_sec'].tolist())
        extract_stops = schedule_df['stop_id'].tolist()
        ordered_stops1.append([str(s) for s in extract_stops])
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
    trip_times2 = [[] for _ in range(nr_intervals)]
    dep_delay2 = [[] for _ in range(delay_nr_intervals)]
    dep_delay_2_ahead = [[] for _ in range(delay_nr_intervals)]
    trip_t2_empirical = [[] for _ in range(nr_intervals)]
    # dep_delay2 = []
    for t in ordered_trip_ids2:
        temp_df = avl_df[avl_df['trip_id'] == t].copy()
        schedule_df = temp_df.drop_duplicates(subset='schd_sec')
        schedule_df = schedule_df.sort_values(by='schd_sec')
        ordered_schedules2.append(schedule_df['schd_sec'].tolist())
        extract_stops = schedule_df['stop_id'].tolist()
        ordered_stops2.append([str(s) for s in extract_stops])
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
                    if dep_delay < tolerance_early_dep:
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
        arr = arr[(arr >= delay1_min) & (arr < delay1_max)]
        arr_ah = np.array(dep_delay_1_ahead[i])
        arr_ah = arr_ah[(arr_ah > delay1_min) & (arr_ah < delay1_max)]
        dep_delay1[i] = list(np.clip(arr, a_min=0, a_max=None))
        dep_delay_1_ahead[i] = list(np.clip(arr_ah, a_min=0, a_max=None))

    # clipping 2 (short) on 0, 200
    delay2_min = -20
    delay2_max = 200
    for i in range(len(dep_delay2)):
        arr = np.array(dep_delay2[i])
        arr = arr[(arr >= delay2_min) & (arr < delay2_max)]
        arr_ah = np.array(dep_delay_2_ahead[i])
        arr_ah = arr_ah[(arr_ah > delay2_min) & (arr_ah < delay2_max)]

        dep_delay2[i] = list(np.clip(arr, a_min=0, a_max=None))
        dep_delay_2_ahead[i] = list(np.clip(arr_ah, a_min=0, a_max=None))

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
    for t_idx in range(2, 4):
        arr1 = remove_outliers(np.array(trip_t1_empirical[t_idx]))
        arr2 = remove_outliers(np.array(trip_t2_empirical[t_idx]))
        trip_t1_empirical[t_idx] = list(arr1)
        trip_t2_empirical[t_idx] = list(arr2)
        axs[t_idx - 2, 0].hist(arr1 / 60, ec='black')
        if arr2.size:
            axs[t_idx - 2, 1].hist(arr2 / 60, ec='black')
    axs[-1, 0].set_xlabel('trip time (min)')
    axs[-1, 1].set_xlabel('trip time (min)')
    axs[0, 0].set_title('long')
    axs[0, 1].set_title('short')
    plt.tight_layout()
    plt.savefig('in/vis/trip_t_in_hist.png')
    plt.close()

    trip_times1_params = []
    trip_times2_params = []
    for i in range(nr_intervals):
        trip_times1[i] = remove_outliers(np.array(trip_times1[i])).tolist()
        trip_times2[i] = remove_outliers(np.array(trip_times2[i])).tolist()

        lognorm_params1 = lognorm.fit(trip_times1[i], floc=0)
        lognorm_params2 = lognorm.fit(trip_times2[i], floc=0)

        trip_times1_params.append(lognorm_params1)
        trip_times2_params.append(lognorm_params2)
    trips1_info = [(x, y, z, w, v) for x, y, z, w, v in
                   zip(ordered_trip_ids1, ordered_deps1, ordered_block_ids1, ordered_schedules1, ordered_stops1)]
    trips2_info = [(x, y, z, w, v) for x, y, z, w, v in
                   zip(ordered_trip_ids2, ordered_deps2, ordered_block_ids2, ordered_schedules2, ordered_stops2)]

    save('in/xtr/trips1_info_inbound.pkl', trips1_info)
    save('in/xtr/trips2_info_inbound.pkl', trips2_info)
    save('in/xtr/trip_time1_params.pkl', trip_times1_params)
    save('in/xtr/trip_time2_params.pkl', trip_times2_params)
    save('in/xtr/dep_delay1_dist_in.pkl', dep_delay1)
    save('in/xtr/dep_delay2_dist_in.pkl', dep_delay2)
    save('in/xtr/trip_t1_dist_in.pkl', trip_t1_empirical)
    save('in/xtr/trip_t2_dist_in.pkl', trip_t2_empirical)
    return


def write_outbound_trajectories(stop_times_path, ordered_trips):
    stop_times_df = pd.read_csv(stop_times_path)
    stop_times_df2 = stop_times_df[stop_times_df['trip_id'].isin(ordered_trips)].copy()
    stop_times_df2 = stop_times_df2.sort_values(by=['stop_sequence', 'schd_sec'])
    stop_times_df2.to_csv('in/vis/trajectories_outbound.csv', index=False)
    return


def get_load_profile(stop_times_path, focus_trips, stops):
    lp = []
    ons_set = []
    offs_set = []
    st_df = pd.read_csv(stop_times_path)
    stop_times_df = st_df[st_df['trip_id'].isin(focus_trips)].copy()
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


def extract_apc_counts(nr_intervals, odt_ordered_stops, interval_len_min, dates):
    arr_rates = np.zeros(shape=(nr_intervals, len(odt_ordered_stops)))
    drop_rates = np.zeros(shape=(nr_intervals, len(odt_ordered_stops)))
    apc_df = pd.read_csv(path_apc_counts)
    for stop_idx in range(len(odt_ordered_stops)):
        print(f'stop {stop_idx + 1}')
        temp_df = apc_df[apc_df['stop_id'] == int(odt_ordered_stops[stop_idx])].copy()
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
