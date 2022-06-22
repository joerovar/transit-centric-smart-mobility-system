from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pickle
from File_Paths import path_stops_out_full_pattern, path_apc_counts, path_avl, path_trips_gtfs, path_stop_times
from File_Paths import path_calendar, path_stops_out_all


def extract_demand(odt_interval_len_min, dates):
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
    full_pattern_stops = load(path_stops_out_full_pattern)
    idx_stops_out = [stops_lst.index(int(s)) for s in full_pattern_stops]
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
                            interval_length, dates, delay_interval_length, delay_start_interval,
                            tolerance_early_departure=1.5 * 60, full_pattern_sign='Illinois Center',
                            rt_nr=20, rt_direction='East'):
    stop_times_df = pd.read_csv(path_stop_times)
    trips_df = pd.read_csv(path_trips_gtfs)
    calendar_df = pd.read_csv(path_calendar)
    rt_trips_df = trips_df[
        (trips_df['route_id'].astype(str) == str(rt_nr)) & (trips_df['direction'] == rt_direction)].copy()
    schd_trip_id_df = rt_trips_df[['trip_id', 'schd_trip_id', 'block_id']].copy()
    schd_trip_id_df['schd_trip_id'] = schd_trip_id_df['schd_trip_id'].astype(int)
    stop_times_df = stop_times_df.merge(schd_trip_id_df, on='trip_id')

    service_ids = calendar_df[(calendar_df['monday'] == 1) & (calendar_df['tuesday'] == 1) &
                              (calendar_df['wednesday'] == 1) & (calendar_df['thursday'] == 1) &
                              (calendar_df['friday'] == 1)]['service_id'].tolist()
    wkday_trip_ids = rt_trips_df[rt_trips_df['service_id'].isin(service_ids)]['trip_id'].tolist()
    wkday_st_df = stop_times_df[stop_times_df['trip_id'].isin(wkday_trip_ids)].copy()
    wkday_st_df = wkday_st_df[wkday_st_df['departure_time'].str[:2] != '24']
    wkday_st_df['schd_sec'] = wkday_st_df['arrival_time'].astype('datetime64[ns]') - pd.to_datetime('00:00:00')
    wkday_st_df['schd_sec'] = wkday_st_df['schd_sec'].dt.total_seconds()

    stop1_st_df = wkday_st_df[wkday_st_df['stop_sequence'] == 1].copy()
    stop1_st_df = stop1_st_df[(stop1_st_df['schd_sec'] >= start_time_sec) &
                              (stop1_st_df['schd_sec'] <= end_time_sec)]
    stop1_st_df = stop1_st_df.sort_values(by='schd_sec')
    trip_ids = stop1_st_df['schd_trip_id'].tolist()
    sched_dep = stop1_st_df['schd_sec'].tolist()
    block_ids = stop1_st_df['block_id'].tolist()

    focus_st_df = wkday_st_df[wkday_st_df['schd_trip_id'].isin(trip_ids)]
    all_stops = focus_st_df['stop_id'].astype(str).unique().tolist()

    full_pattern_df = wkday_st_df[wkday_st_df['stop_headsign'] == full_pattern_sign].copy()
    full_pattern_df = full_pattern_df.drop_duplicates(subset='stop_sequence')
    full_pattern_df = full_pattern_df.sort_values(by='stop_sequence')
    full_pattern_stops = full_pattern_df['stop_id'].astype(str).tolist()

    focus_st_df = focus_st_df.sort_values(by='schd_sec')
    stop_info = {}
    for s in all_stops:
        stop_info[s] = focus_st_df[focus_st_df['stop_id'].astype(str) == s]['schd_sec'].tolist()
    focus_st_df.to_csv('in/vis/stop_times_shrink.txt', index=False)

    avl_df = pd.read_csv(path_avl)
    schedules_by_trip = []
    stops_by_trip = []
    dist_by_trip = []

    link_times = {}

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)
    dep_delay_dist = [[] for _ in range(delay_nr_intervals)]
    dep_delay_ahead_dist = [[] for _ in range(delay_nr_intervals)]

    write_outbound_trajectories(path_avl, trip_ids)
    for t in trip_ids:
        temp2 = focus_st_df[focus_st_df['schd_trip_id'] == t].copy()
        temp2 = temp2.sort_values(by='stop_sequence')
        schedules_by_trip.append(temp2['schd_sec'].tolist())
        stops_by_trip.append(temp2['stop_id'].astype(str).tolist())
        dist_by_trip.append(temp2['shape_dist_traveled'].tolist())

        temp = avl_df[avl_df['trip_id'] == t].copy()
        temp = temp.sort_values(by='stop_sequence')

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
                        if link not in link_times:
                            link_times[link] = [[] for _ in range(nr_intervals)]
                        nr_bin = get_interval(avl_sec[i] % 86400, interval_length) - start_interval
                        if 0 <= nr_bin < nr_intervals:
                            lt2 = avl_sec[i + 1] - avl_dep_sec[i]
                            if lt2 > 0:
                                link_times[link][nr_bin].append(lt2)
    delay_max = 300
    for i in range(delay_nr_intervals):
        dep_delay_ahead_dist[i] = list(remove_outliers(np.array(dep_delay_ahead_dist[i])))
        arr = np.array(dep_delay_ahead_dist[i])
        dep_delay_ahead_dist[i] = list(arr[arr<=delay_max])

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
                    new_dist = lognorm.rvs(*fit_params, size=30)
                    new_mean = new_dist.mean()
                    new_cv = np.std(new_dist) / new_mean

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

    trips_info = [(v, w, x, y, z, u) for v, w, x, y, z, u in
                  zip(trip_ids, sched_dep, block_ids, schedules_by_trip, stops_by_trip, dist_by_trip)]

    stop_df = pd.read_csv('in/raw/gtfs/stops.txt')
    stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in all_stops])]
    stop_df = stop_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
    stop_df.to_csv('in/raw/rt20_in_stops.txt', index=False)

    save(path_stops_out_full_pattern, full_pattern_stops)
    save(path_stops_out_all, all_stops)
    save('in/xtr/link_times_info.pkl', link_times_info)
    save('in/xtr/trips_outbound_info.pkl', trips_info)
    save('in/xtr/dep_delay_dist_out.pkl', dep_delay_ahead_dist)
    save('in/xtr/stops_out_info.pkl', stop_info)
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
                           delay_start_interval, rt_nr=20, rt_direction='West'):
    stop_times_df = pd.read_csv(path_stop_times)
    trips_df = pd.read_csv(path_trips_gtfs)
    calendar_df = pd.read_csv(path_calendar)
    rt_trips_df = trips_df[
        (trips_df['route_id'].astype(str) == str(rt_nr)) & (trips_df['direction'] == rt_direction)].copy()
    schd_trip_id_df = rt_trips_df[['trip_id', 'schd_trip_id', 'block_id']].copy()
    schd_trip_id_df['schd_trip_id'] = schd_trip_id_df['schd_trip_id'].astype(int)
    stop_times_df = stop_times_df.merge(schd_trip_id_df, on='trip_id')

    service_ids = calendar_df[(calendar_df['monday'] == 1) & (calendar_df['tuesday'] == 1) &
                              (calendar_df['wednesday'] == 1) & (calendar_df['thursday'] == 1) &
                              (calendar_df['friday'] == 1)]['service_id'].tolist()
    wkday_trip_ids = rt_trips_df[rt_trips_df['service_id'].isin(service_ids)]['trip_id'].tolist()
    wkday_st_df = stop_times_df[stop_times_df['trip_id'].isin(wkday_trip_ids)].copy()
    wkday_st_df = wkday_st_df[wkday_st_df['departure_time'].str[:2] != '24']
    wkday_st_df['schd_sec'] = wkday_st_df['arrival_time'].astype('datetime64[ns]') - pd.to_datetime('00:00:00')
    wkday_st_df['schd_sec'] = wkday_st_df['schd_sec'].dt.total_seconds()

    stop1_st_df = wkday_st_df[wkday_st_df['stop_sequence'] == 1].copy()
    stop1_st_df = stop1_st_df[(stop1_st_df['schd_sec'] >= start_time) &
                              (stop1_st_df['schd_sec'] <= end_time)]
    stop1_st_df = stop1_st_df.sort_values(by='schd_sec')

    trip_ids = stop1_st_df['schd_trip_id'].tolist()
    sched_dep = stop1_st_df['schd_sec'].tolist()
    block_ids = stop1_st_df['block_id'].tolist()

    sched_by_trip = []
    stops_by_trip = []
    dist_by_trip = []


    avl_df = pd.read_csv(path_avl)

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)

    run_times = {}
    dep_delay_new = {}
    for t in trip_ids:
        temp2 = wkday_st_df[wkday_st_df['schd_trip_id'] == t].copy()
        temp2 = temp2.sort_values(by='stop_sequence')
        schd_sec = temp2['schd_sec'].tolist()
        stop_seq = temp2['stop_sequence'].tolist()
        stop_ids = temp2['stop_id'].astype(str).tolist()
        dist_traveled = temp2['shape_dist_traveled'].tolist()
        sched_by_trip.append(schd_sec)
        stops_by_trip.append(stop_ids)
        dist_by_trip.append(dist_traveled)

        temp_df = avl_df[avl_df['trip_id'] == t].copy()
        trip_link = stop_ids[0] + '-' + stop_ids[-1]
        if trip_link not in run_times:
            run_times[trip_link] = [[] for _ in range(nr_intervals)]
        if stop_ids[0] not in dep_delay_new:
            dep_delay_new[stop_ids[0]] = [[] for _ in range(delay_nr_intervals)]
        delay_idx = get_interval(schd_sec[0], delay_interval_length) - delay_start_interval
        run_t_idx = get_interval(schd_sec[0], interval_length) - start_interval
        for d in dates:
            df = temp_df[temp_df['avl_arr_time'].astype(str).str[:10] == d].copy()
            s1 = df[df['stop_sequence'] == 1]
            sm = df[df['stop_sequence'] == 2]
            sm2 = df[df['stop_sequence'] == stop_seq[-2]]
            s2 = df[df['stop_sequence'] == stop_seq[-1]]
            if not s1.empty and not s2.empty and not sm.empty and not sm2.empty:
                s1_id = s1['stop_id'].iloc[0].astype(str)
                s2_id = s2['stop_id'].iloc[0].astype(str)
                link = s1_id + '-' + s2_id
                assert link == stop_ids[0] + '-' + stop_ids[-1]
                t1 = s1['avl_dep_sec'].iloc[0] % 86400
                t2 = s2['avl_arr_sec'].iloc[0] % 86400
                tm = sm['avl_dep_sec'].iloc[0] % 86400
                tm2 = sm2['avl_dep_sec'].iloc[0] % 86400
                dep_del = t1 - schd_sec[0]
                dep_delay_new[stop_ids[0]][delay_idx].append(dep_del)
                run_t = (tm2 - tm) + (schd_sec[1]-schd_sec[0]) + (schd_sec[-1]-schd_sec[-2])
                run_times[link][run_t_idx].append(run_t)
    for link in run_times:
        fig, axs = plt.subplots(nrows=nr_intervals, sharex='all', sharey='all')
        plt.suptitle(link)
        for interval in range(nr_intervals):
            # axs[interval].hist(run_times[link][interval], density=True)
            run_times[link][interval] = list(remove_outliers(np.array(run_times[link][interval])))
            axs[interval].hist(run_times[link][interval], density=True)
        plt.show()
        plt.close()
    for stop in dep_delay_new:
        fig, axs = plt.subplots(nrows=nr_intervals, sharex='all', sharey='all')
        plt.suptitle(stop)
        for interval in range(delay_nr_intervals):
            # axs[interval, 0].hist(dep_delay_new[stop][interval], density=True)
            dep_delay_new[stop][interval] = list(remove_outliers(np.array(dep_delay_new[stop][interval])))
            axs[interval].hist(dep_delay_new[stop][interval], density=True)
        plt.show()
        plt.close()

    trips_info = [(x, y, z, w, v, u) for x, y, z, w, v, u in
                  zip(trip_ids, sched_dep, block_ids, sched_by_trip, stops_by_trip, dist_by_trip)]

    save('in/xtr/trips_inbound_info.pkl', trips_info)
    save('in/xtr/run_times_in.pkl', run_times)
    save('in/xtr/delay_in.pkl', dep_delay_new)
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
