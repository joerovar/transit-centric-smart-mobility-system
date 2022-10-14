import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pickle
from ins.Fixed_Inputs_81 import DIR_ROUTE, DIR_ROUTE_OUTS


def extract_demand(odt_interval_len_min, dates, apc_counts_file, apc_on_rates=None, apc_off_rates=None):
    odt_stops = np.load(DIR_ROUTE + 'odt_stops.npy')
    odt_pred = np.load(DIR_ROUTE + 'odt_flows_30.npy')
    wkday_trip_ids = np.load(DIR_ROUTE + 'wkday_trip_ids_out.npy').tolist() + np.load(DIR_ROUTE + 'wkday_trip_ids_in.npy').tolist()
    wkday_schd_trip_ids = np.load(DIR_ROUTE + 'wkday_schd_trip_ids_out.npy').tolist() + np.load(DIR_ROUTE + 'wkday_schd_trip_ids_in.npy').tolist()
    stop_t_df = pd.read_csv(DIR_ROUTE + 'gtfs_stop_times.txt')
    stop_t_df = stop_t_df[stop_t_df['arrival_time'].str[:2] != '24'].copy()
    stop_t_df['arr_sec'] = (pd.to_datetime(stop_t_df['arrival_time'])- pd.to_datetime('2022-05-02 00:00:00')).dt.total_seconds()%86400
    nr_intervals = 24 / (odt_interval_len_min / 60)
    apc_df = pd.read_csv(DIR_ROUTE + apc_counts_file)
    if apc_on_rates is None or apc_off_rates is None:
        apc_on_rates, apc_off_rates = extract_apc_counts(apc_df, int(nr_intervals), odt_stops, odt_interval_len_min,
                                                         dates, wkday_schd_trip_ids, stop_t_df, wkday_trip_ids)
        print('resulting apc on')
        print(apc_on_rates)
    scaled_odt = np.zeros_like(odt_pred)

    for i in range(odt_pred.shape[0]):
        print(f'interval {i}')
        scaled_odt[i] = bi_proportional_fitting(odt_pred[i], apc_on_rates[i], apc_off_rates[i])

    np.save(DIR_ROUTE + 'odt_flows_30_scaled.npy', scaled_odt)

    # if wanted for comparison
    stops_lst = list(odt_stops)
    full_pattern_stops = load(DIR_ROUTE + 'stops_out_full_patt.pkl')
    idx_stops_out = [stops_lst.index(str(s)) for s in full_pattern_stops]
    out_on_counts = apc_on_rates[:, idx_stops_out]
    out_on_tot_count = np.nansum(out_on_counts, axis=-1)

    scaled_arr_rates = np.nansum(scaled_odt, axis=-1)
    scaled_out_arr_rates = scaled_arr_rates[:, idx_stops_out]
    scaled_out_tot = np.nansum(scaled_out_arr_rates, axis=-1)

    unscaled_arr_rates = np.nansum(odt_pred, axis=-1)
    unscaled_out_arr_rates = unscaled_arr_rates[:, idx_stops_out]
    unscaled_out_tot = np.nansum(unscaled_out_arr_rates, axis=-1)

    x = np.arange(out_on_tot_count.shape[0])
    plt.plot(x, scaled_out_tot, label='odt scaled')
    plt.plot(x, out_on_tot_count, label='apc')
    plt.plot(x, unscaled_out_tot, label='odt not scaled')
    plt.xticks(np.arange(0, out_on_tot_count.shape[0], 2), np.arange(int(out_on_tot_count.shape[0] / 2)))
    plt.title('boardings per hour (outbound)')
    plt.xlabel('hour of day')
    plt.ylabel('boardings per hour')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(DIR_ROUTE_OUTS + 'compare/validate/odt_scaling.jpg')
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
    # t is ins seconds and len_i ins minutes
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


def extract_outbound_params(start_time_sec, end_time_sec, nr_intervals, start_interval, interval_length, 
                            dates, delay_interval_length, delay_start_interval, full_pattern_sign,
                            rt_nr, rt_direction, min_speed=3, max_speed=25):
    stop_times_df = pd.read_csv(DIR_ROUTE + 'gtfs_stop_times.txt')
    trips_df = pd.read_csv(DIR_ROUTE + 'gtfs_trips.txt')
    calendar_df = pd.read_csv(DIR_ROUTE + 'gtfs_calendar.txt')
    rt_trips_df = trips_df[
        (trips_df['route_id'].astype(str) == str(rt_nr)) & (trips_df['direction'] == rt_direction)].copy()
    schd_trip_id_df = rt_trips_df[['trip_id', 'schd_trip_id', 'block_id']].copy()
    schd_trip_id_df['schd_trip_id'] = schd_trip_id_df['schd_trip_id'].astype(int)
    stop_times_df = stop_times_df.merge(schd_trip_id_df, on='trip_id')

    wkday_calendar_df = calendar_df[(calendar_df['monday'] == 1) & (calendar_df['tuesday'] == 1) &
                              (calendar_df['wednesday'] == 1) & (calendar_df['thursday'] == 1) &
                              (calendar_df['friday'] == 1)].copy()
    wkday_calendar_df['start_dt'] = pd.to_datetime(wkday_calendar_df['start_date'].astype(str))
    wkday_calendar_df['end_dt'] = pd.to_datetime(wkday_calendar_df['end_date'].astype(str))
    wkday_calendar_df = wkday_calendar_df[(wkday_calendar_df['start_dt']<=pd.to_datetime(dates[0])) & 
                                        (wkday_calendar_df['end_dt']>=pd.to_datetime(dates[-1]))]
    service_ids = wkday_calendar_df['service_id'].astype(str).tolist()
    wkday_trips_df = rt_trips_df[rt_trips_df['service_id'].astype(str).isin(service_ids)].copy()
    wkday_trip_ids = wkday_trips_df['trip_id'].tolist()
    wkday_schd_trip_ids = wkday_trips_df['schd_trip_id'].tolist()

    wkday_st_df = stop_times_df[stop_times_df['trip_id'].isin(wkday_trip_ids)].copy()
    wkday_st_df = wkday_st_df[wkday_st_df['departure_time'].str[:2] != '24'].copy()
    wkday_st_df['schd_sec'] = wkday_st_df['arrival_time'].astype('datetime64[ns]') - pd.to_datetime('00:00:00')
    wkday_st_df['schd_sec'] = wkday_st_df['schd_sec'].dt.total_seconds()
    wkday_st_df.to_pickle(DIR_ROUTE + 'schedule_df.pkl')

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
    # focus_st_df.to_csv('ins/vis/stop_times_shrink.txt', index=False)

    avl_df = pd.read_csv(DIR_ROUTE + 'avl.csv')
    schedules_by_trip = []
    stops_by_trip = []
    dist_by_trip = []

    link_times = {}

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)
    dep_delay_ahead_dist = [[] for _ in range(delay_nr_intervals)]

    for t in trip_ids:
        temp2 = focus_st_df[focus_st_df['schd_trip_id'] == t].copy()
        temp2 = temp2.sort_values(by='stop_sequence')
        schedules_by_trip.append(temp2['schd_sec'].tolist())
        stops_by_trip.append(temp2['stop_id'].astype(str).tolist())
        dist_by_trip.append(temp2['shape_dist_traveled'].tolist())
        dist_df = temp2[['stop_id', 'shape_dist_traveled']].copy()
        dist_df['dist'] = dist_df['shape_dist_traveled'].diff().shift(-1)
        dist_df['next_stop_id'] = dist_df['stop_id'].astype(str).shift(-1)
        dist_df['link'] = dist_df['stop_id'].astype(str) + '-' + dist_df['next_stop_id']
        dist_df = dist_df[['link', 'dist']]
        trip_df = avl_df[avl_df['trip_id'] == t].copy()

        for d in dates:
            temp = trip_df[trip_df['arr_time'].astype(str).str[:10] == d].copy()
            temp = temp.sort_values(by='stop_sequence')
            temp['seq_diff'] = temp['stop_sequence'].diff().shift(-1)
            temp['next_stop_id'] = temp['stop_id'].astype(str).shift(-1)
            temp['next_arr_sec'] = temp['arr_sec'].shift(-1)
            temp = temp.dropna(subset=['seq_diff'])
            temp = temp[temp['seq_diff'] == 1.0]
            temp['link'] = temp['stop_id'].astype(str) + '-' + temp['next_stop_id']
            temp['link_t'] = temp['next_arr_sec'] % 86400 - temp['dep_sec'] % 86400
            temp = temp[temp['link_t'] > 0.0].copy()

            temp = temp.merge(dist_df, on='link')
            temp['speed_mph'] =  temp['dist'] / temp['link_t'] * 0.682
            # if temp.shape[0]:
            #     print(f'---')
            #     print(f'before {temp.shape[0]}')
            temp = temp[(temp['speed_mph'] <= max_speed) & (temp['speed_mph'] >= min_speed)].copy()
            # if temp.shape[0]:
            #     print(f'after {temp.shape[0]}')
            temp_links = temp['link'].tolist()
            temp_link_t = temp['link_t'].tolist()
            temp_start_sec = temp['dep_sec'].tolist()
            stop_2 = temp[temp['stop_sequence'] == 2].copy()
            if not stop_2.empty:
                stop_2_schd_sec = stop_2['schd_sec'].iloc[0]
                stop_2_avl_sec = stop_2['dep_sec'].iloc[0] % 86400
                delay_interv_idx = get_interval(stop_2_schd_sec, delay_interval_length) - delay_start_interval
                dep_delay_ahead_dist[delay_interv_idx].append(stop_2_avl_sec - stop_2_schd_sec)
            for i in range(len(temp_links)):
                interval_idx = get_interval(temp_start_sec[i], interval_length) - start_interval
                link = temp_links[i]
                if link not in link_times:
                    link_times[link] = [[] for _ in range(nr_intervals)]
                if 0 <= interval_idx < nr_intervals:
                    link_times[link][interval_idx].append(temp_link_t[i])
    delay_max = 360
    for i in range(delay_nr_intervals):
        dep_delay_ahead_dist[i] = list(remove_outliers(np.array(dep_delay_ahead_dist[i])))
        arr = np.array(dep_delay_ahead_dist[i])
        dep_delay_ahead_dist[i] = list(arr[arr <= delay_max])

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

    stop_df = pd.read_csv(DIR_ROUTE + 'gtfs_stops.txt')
    stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in all_stops])]
    stop_df = stop_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
    stop_df['stop_id'] = stop_df['stop_id'].astype(str).astype('category')
    stop_df['stop_id'].cat.set_categories(full_pattern_stops, inplace=True)
    stop_df = stop_df.sort_values(by='stop_id')
    stop_df['short_name'] = stop_df['stop_name'].str.split(' ', n=2, expand=True)[2].str.upper()
    stop_df['stop_id'] = stop_df['stop_id'].astype(str)
    stop_df.to_csv(DIR_ROUTE + 'gtfs_stops_route.txt', index=False)
    np.save(DIR_ROUTE + 'wkday_schd_trip_ids_out.npy', wkday_schd_trip_ids)
    np.save(DIR_ROUTE + 'wkday_trip_ids_out.npy', wkday_trip_ids)
    save(DIR_ROUTE + 'stops_out_full_patt.pkl', full_pattern_stops)
    save(DIR_ROUTE + 'stops_out_all.pkl', all_stops)
    save(DIR_ROUTE + 'link_times_info.pkl', link_times_info)
    save(DIR_ROUTE + 'trips_out_info.pkl', trips_info)
    save(DIR_ROUTE + 'delay_distr_out.pkl', dep_delay_ahead_dist)
    save(DIR_ROUTE + 'stops_out_info.pkl', stop_info)
    return


def bi_proportional_fitting(od, target_ons, target_offs):
    balance_target_factor = np.sum(target_ons) / np.sum(target_offs)
    balanced_target_offs = target_offs * balance_target_factor
    od_temp = od.copy()
    print('before')
    print(np.round(np.nansum(od)), np.round(np.nansum(target_ons)), np.round(np.nansum(od)))
    for i in range(15):
        # balance rows
        actual_ons = np.nansum(od_temp, axis=1)
        factor_ons = np.divide(target_ons, actual_ons, out=np.zeros_like(target_ons), where=actual_ons != 0)
        od_temp = od_temp * factor_ons[:, np.newaxis]
        actual_ons = np.nansum(od_temp, axis=1)
        # balance columns
        actual_offs = np.nansum(od_temp, axis=0)
        factor_offs = np.divide(balanced_target_offs, actual_offs, out=np.zeros_like(target_offs),
                                where=actual_offs != 0)
        od_temp = od_temp * factor_offs
        actual_offs = np.nansum(od_temp, axis=0)
        # to check for tolerance we first assign 1.0 to totals of zero which cannot be changed by the method
        factor_ons[actual_ons == 0] = 1.0
        factor_offs[actual_offs == 0] = 1.0

        target = target_ons[actual_ons == 0]
    print('after')
    print(np.round(np.nansum(od_temp)), np.round(np.nansum(target_ons)), np.round(np.nansum(od)))
    # print(factor_ons)
    scaled_od_set = np.array(od_temp)
    return scaled_od_set


def extract_inbound_params(start_time, end_time, dates, nr_intervals,
                           start_interval, interval_length, delay_interval_length,
                           delay_start_interval, rt_nr, rt_direction):
    stop_times_df = pd.read_csv(DIR_ROUTE + 'gtfs_stop_times.txt')
    trips_df = pd.read_csv(DIR_ROUTE + 'gtfs_trips.txt')
    calendar_df = pd.read_csv(DIR_ROUTE + 'gtfs_calendar.txt')
    avl_df = pd.read_csv(DIR_ROUTE + 'avl.csv')

    rt_trips_df = trips_df[
        (trips_df['route_id'].astype(str) == str(rt_nr)) & (trips_df['direction'] == rt_direction)].copy()
    schd_trip_id_df = rt_trips_df[['trip_id', 'schd_trip_id', 'block_id']].copy()
    schd_trip_id_df['schd_trip_id'] = schd_trip_id_df['schd_trip_id'].astype(int)
    stop_times_df = stop_times_df.merge(schd_trip_id_df, on='trip_id')

    wkday_calendar_df = calendar_df[(calendar_df['monday'] == 1) & (calendar_df['tuesday'] == 1) &
                              (calendar_df['wednesday'] == 1) & (calendar_df['thursday'] == 1) &
                              (calendar_df['friday'] == 1)].copy()
    wkday_calendar_df['start_dt'] = pd.to_datetime(wkday_calendar_df['start_date'].astype(str))
    wkday_calendar_df['end_dt'] = pd.to_datetime(wkday_calendar_df['end_date'].astype(str))
    wkday_calendar_df = wkday_calendar_df[(wkday_calendar_df['start_dt']<=pd.to_datetime(dates[0])) & 
                                        (wkday_calendar_df['end_dt']>=pd.to_datetime(dates[-1]))]
    service_ids = wkday_calendar_df['service_id'].astype(str).tolist()
    wkday_trips_df = rt_trips_df[rt_trips_df['service_id'].astype(str).isin(service_ids)].copy()
    wkday_trip_ids = wkday_trips_df['trip_id'].tolist()
    wkday_schd_trip_ids = wkday_trips_df['schd_trip_id'].tolist()
    np.save(DIR_ROUTE + 'wkday_schd_trip_ids_in.npy', wkday_schd_trip_ids)
    np.save(DIR_ROUTE + 'wkday_trip_ids_in.npy', wkday_trip_ids)
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

    delay_nr_intervals = int(nr_intervals * interval_length / delay_interval_length)

    run_times = {}
    dep_delay = {}

    diff_tt = {}
    diff_mm = {}
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
        sched_run_t = schd_sec[-1] - schd_sec[0]
        if trip_link not in run_times:
            run_times[trip_link] = [[] for _ in range(nr_intervals)]
            diff_mm[trip_link] = []
            diff_tt[trip_link] = []
        if stop_ids[0] not in dep_delay:
            dep_delay[stop_ids[0]] = [[] for _ in range(delay_nr_intervals)]
        delay_idx = get_interval(schd_sec[0], delay_interval_length) - delay_start_interval
        run_t_idx = get_interval(schd_sec[0], interval_length) - start_interval

        for d in dates:
            df = temp_df[temp_df['arr_time'].astype(str).str[:10] == d].copy()
            sm1 = df[df['stop_sequence'] == 2]
            sm2 = df[df['stop_sequence'] == stop_seq[-2]]

            if not sm1.empty and not sm2.empty:
                sm1_id = sm1['stop_id'].iloc[0].astype(str)
                sm2_id = sm2['stop_id'].iloc[0].astype(str)
                assert sm1_id + '-' + sm2_id == stop_ids[1] + '-' + stop_ids[-2]
                tm1 = sm1['dep_sec'].iloc[0] % 86400
                tm2 = sm2['arr_sec'].iloc[0] % 86400
                dep_del = tm1 - schd_sec[1]
                dep_delay[stop_ids[0]][delay_idx].append(dep_del)
                run_t = (tm2 - tm1) + (schd_sec[1] - schd_sec[0]) + (schd_sec[-1] - schd_sec[-2])
                run_times[trip_link][run_t_idx].append(run_t)

    for link in run_times:
        fig, axs = plt.subplots(nrows=nr_intervals, sharex='all', sharey='all')
        plt.suptitle(link)
        for interval in range(nr_intervals):
            # axs[interval].hist(run_times[link][interval], density=True)
            run_times[link][interval] = list(remove_outliers(np.array(run_times[link][interval])))
            axs[interval].hist(run_times[link][interval], density=True)
        # plt.show()
        plt.close()
    for stop in dep_delay:
        fig, axs = plt.subplots(nrows=nr_intervals, sharex='all', sharey='all')
        plt.suptitle(stop)
        for interval in range(delay_nr_intervals):
            # axs[interval, 0].hist(dep_delay_new[stop][interval], density=True)
            dep_delay[stop][interval] = list(remove_outliers(np.array(dep_delay[stop][interval])))
            axs[interval].hist(dep_delay[stop][interval], density=True)
        # plt.show()
        plt.close()

    trips_info = [(x, y, z, w, v, u) for x, y, z, w, v, u in
                  zip(trip_ids, sched_dep, block_ids, sched_by_trip, stops_by_trip, dist_by_trip)]

    save(DIR_ROUTE + 'trips_in_info.pkl', trips_info)
    save(DIR_ROUTE + 'run_t_distr_in.pkl', run_times)
    save(DIR_ROUTE + 'delay_distr_in.pkl', dep_delay)
    return


def extract_apc_counts(apc_df, nr_bins, odt_ordered_stops, interval_len_min, dates, schd_trip_ids,
                        stop_t_df, trip_ids):
    arr_rates = np.zeros(shape=(nr_bins, len(odt_ordered_stops)))
    drop_rates = np.zeros(shape=(nr_bins, len(odt_ordered_stops)))
    for stop_idx in range(len(odt_ordered_stops)):
        print(f'stop {stop_idx + 1}')
        temp_df = apc_df[apc_df['stop_id'] == int(odt_ordered_stops[stop_idx])].copy()
        temp_df = temp_df[temp_df['trip_id'].isin(schd_trip_ids)].copy()
        stop_df = stop_t_df[stop_t_df['stop_id']==int(odt_ordered_stops[stop_idx])].copy()
        stop_df = stop_df[stop_df['trip_id'].isin(trip_ids)].copy()
        for bin_nr in range(48):
            t_edge0 = bin_nr * interval_len_min * 60
            t_edge1 = (bin_nr + 1) * interval_len_min * 60
            pax_df = temp_df[(temp_df['arr_sec'] % 86400 <= t_edge1) & (temp_df['arr_sec'] % 86400 >= t_edge0)].copy()
            schd_df = stop_df[(stop_df['arr_sec'] % 86400 >= t_edge0) & (stop_df['arr_sec'] % 86400 <= t_edge1)].copy()
            n_trips = schd_df.shape[0]
            on_rates = []
            off_rates = []
            for k in range(len(dates)):
                day_df = pax_df[pax_df['arr_time'].astype(str).str[:10] == dates[k]].copy()
                if not day_df.empty:
                    if day_df.shape[0] >= n_trips-1:
                        on_rates.append((day_df['fon'].sum() + day_df['ron'].sum()) * 60 / interval_len_min)
                        off_rates.append((day_df['foff'].sum() + day_df['roff'].sum()) * 60 / interval_len_min)
            if on_rates:
                arr_rates[bin_nr, stop_idx] = np.mean(on_rates)
                drop_rates[bin_nr, stop_idx] = np.mean(off_rates)
    np.save(DIR_ROUTE + 'apc_on_rates_30.npy', arr_rates)
    np.save(DIR_ROUTE + 'apc_off_rates_30.npy', drop_rates)
    return arr_rates, drop_rates


