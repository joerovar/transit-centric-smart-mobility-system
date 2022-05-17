from pre_process import *
from file_paths import *
from constants import *
from post_process import save, load
from datetime import timedelta


def extract_params(outbound_route_params=False, inbound_route_params=False, demand=False, validation=False):
    if outbound_route_params:
        stops, trips_out_info, link_times_info, sched_arrs_out, dep_delay_out = get_route(path_stop_times,
                                                                                          START_TIME_SEC, END_TIME_SEC,
                                                                                          TIME_NR_INTERVALS,
                                                                                          TIME_START_INTERVAL,
                                                                                          TIME_INTERVAL_LENGTH_MINS,
                                                                                          DATES,
                                                                                          TRIP_WITH_FULL_STOP_PATTERN,
                                                                                          path_avl,
                                                                                          DELAY_INTERVAL_LENGTH_MINS,
                                                                                          DELAY_START_INTERVAL)
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_info)
        save('in/xtr/trips_outbound_info.pkl', trips_out_info)
        save('in/xtr/scheduled_arrivals_outbound.pkl', sched_arrs_out)
        save('in/xtr/dep_delay_dist_out.pkl', dep_delay_out)
        stop_df = pd.read_csv('in/raw/gtfs/stops.txt')
        stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in stops])]

        stop_seq_dict = {'stop_id': [int(s) for s in stops], 'stop_seq': [i for i in range(1, len(stops) + 1)]}
        stop_seq_df = pd.DataFrame(stop_seq_dict)
        stop_df = pd.merge(stop_df, stop_seq_df, on='stop_id')
        stop_df = stop_df.sort_values(by='stop_seq')
        stop_df = stop_df[['stop_seq', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        stop_df.to_csv('in/raw/rt20_in_stops.txt', index=False)

    if inbound_route_params:
        trips1_info_out, trips2_info_out, sched_arrs_out, trip_times1_params, trip_times2_params,\
            delay1_dist, delay2_dist, trip_t1_dist, trip_t2_dist = get_inbound_travel_time(
            path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
            TRIP_TIME_INTERVAL_LENGTH_MINS, path_avl, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL)
        save('in/xtr/trips1_info_inbound.pkl', trips1_info_out)
        save('in/xtr/trips2_info_inbound.pkl', trips2_info_out)
        save('in/xtr/scheduled_arrivals_inbound.pkl', sched_arrs_out)
        save('in/xtr/trip_time1_params.pkl', trip_times1_params)
        save('in/xtr/trip_time2_params.pkl', trip_times2_params)
        save('in/xtr/dep_delay1_dist_in.pkl', delay1_dist)
        save('in/xtr/dep_delay2_dist_in.pkl', delay2_dist)
        save('in/xtr/trip_t1_dist_in.pkl', trip_t1_dist)
        save('in/xtr/trip_t2_dist_in.pkl', trip_t2_dist)
        # save('in/xtr/deadhead_times_params.pkl', deadhead_time_params)

    if demand:
        stops_outbound = load(path_route_stops)
        odt_stops = np.load('in/xtr/rt_20_odt_stops.npy')
        # comes from project with dingyi data
        odt_pred = np.load('in/xtr/rt_20_odt_rates_30.npy')
        # comes from project with dingyi data

        # nr_intervals = 24/(ODT_INTERVAL_LEN_MIN/60)
        # apc_on_rates, apc_off_rates = extract_apc_counts(nr_intervals, odt_stops, path_stop_times, ODT_INTERVAL_LEN_MIN, DATES)
        # save('in/xtr/apc_on_counts.pkl', apc_on_rates)
        # save('in/xtr/apc_off_counts.pkl', apc_off_rates)
        apc_on_rates = load('in/xtr/apc_on_counts.pkl')
        apc_off_rates = load('in/xtr/apc_off_counts.pkl')
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
        prev_odt = load('in/xtr/odt.pkl')
        out_on_counts = apc_on_rates[:, idx_stops_out]
        out_on_tot_count = np.nansum(out_on_counts, axis=-1)

        arr_rates_shifted = np.nansum(shifted_odt, axis=-1)
        out_arr_rates_shifted = arr_rates_shifted[:, idx_stops_out]
        out_arr_tot_shifted = np.sum(out_arr_rates_shifted, axis=-1)

        scaled_arr_rates = np.sum(scaled_odt, axis=-1)
        scaled_out_arr_rates = scaled_arr_rates[:, idx_stops_out]
        scaled_out_tot = np.sum(scaled_out_arr_rates, axis=-1)

        arr_rates_old = np.nansum(prev_odt, axis=-1)
        old_out_tot = np.nansum(arr_rates_old, axis=-1)

        x = np.arange(out_on_tot_count.shape[0])
        plt.plot(np.arange(10, 20, 2), old_out_tot, label='previous odt')
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

    if validation:
        stops = load(path_route_stops)
        trips_outbound_info = load('in/xtr/trips_outbound_info.pkl')
        scheduled_dep_in, ordered_trips_in = [], []
        for t in trips_outbound_info:
            scheduled_dep_in.append(t[1]), ordered_trips_in.append(t[0])
        ordered_trips_in = np.array(ordered_trips_in)
        schedule_arr = np.array(scheduled_dep_in)
        focus_trips = ordered_trips_in[
            (schedule_arr <= FOCUS_END_TIME_SEC) & (schedule_arr >= FOCUS_START_TIME_SEC)].tolist()
        trip_times, headway_out, headway_out_cv, hw_out_cv2 = get_trip_times(path_avl, focus_trips, DATES, stops)
        save('in/xtr/trip_t_outbound.pkl', trip_times)
        save('in/xtr/departure_headway_outbound.pkl', headway_out)
        save('in/xtr/cv_hw_outbound.pkl', headway_out_cv)
        save('in/xtr/cv_hw_out2.pkl', hw_out_cv2)
        load_profile, ons, offs = get_load_profile(path_stop_times, focus_trips, stops)
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.plot(load_profile)
        x = np.arange(len(load_profile))
        w = 0.5
        ax1.bar(x, ons, w)
        ax1.bar(x + w, offs, w)
        plt.savefig('in/vis/pax_profile_observed.png')
        plt.close()
        save('in/xtr/load_profile.pkl', load_profile)

        dep_delay_dist_out = load('in/xtr/dep_delay_dist_out.pkl')
        dep_delay1_dist_in = load('in/xtr/dep_delay1_dist_in.pkl')
        dep_delay2_dist_in = load('in/xtr/dep_delay2_dist_in.pkl')
        dep_delay_dist_out_samp = [[] for _ in range(len(dep_delay_dist_out))]
        dep_delay1_dist_in_samp = [[] for _ in range(len(dep_delay1_dist_in))]
        dep_delay2_dist_in_samp = [[] for _ in range(len(dep_delay2_dist_in))]

        n_samples = 30
        for i in range(len(dep_delay_dist_out)):
            sample_percentiles = np.random.uniform(low=0.0, high=100.0, size=n_samples)
            if dep_delay_dist_out[i]:
                dep_delay_dist_out_samp[i] = list(np.percentile(dep_delay_dist_out[i], sample_percentiles))
            if dep_delay1_dist_in[i]:
                dep_delay1_dist_in_samp[i] = list(np.percentile(dep_delay1_dist_in[i], sample_percentiles))
            if dep_delay2_dist_in[i]:
                dep_delay2_dist_in_samp[i] = list(np.percentile(dep_delay2_dist_in[i], sample_percentiles))
        fig, axs = plt.subplots(nrows=2, ncols=3, sharey='all', sharex='col')
        for i in range(2, 4):
            axs[i - 2, 0].hist([dep_delay_dist_out[i], dep_delay_dist_out_samp[i]], density=True,
                               label=['historical', 'sampled'])
            axs[i - 2, 1].hist([dep_delay1_dist_in[i], dep_delay1_dist_in_samp[i]], density=True,
                               label=['historical', 'sampled'])
            axs[i - 2, 2].hist([dep_delay2_dist_in[i], dep_delay2_dist_in_samp[i]], density=True,
                               label=['historical', 'sampled'])
        axs[0, 0].set_title('outbound')
        axs[0, 1].set_title('inbound long')
        axs[0, 2].set_title('inbound short')
        plt.legend()
        plt.savefig('out/compare/validate/initial_delay.png')
        plt.close()

    return


def get_params_outbound():
    stops_out = load(path_route_stops)
    link_times_info = load(path_link_times_mean)
    trips_out_info = load('in/xtr/trips_outbound_info.pkl')
    odt_rates_old = load(path_odt_rates_xtr)
    sched_arrivals = load('in/xtr/scheduled_arrivals_outbound.pkl')
    odt_rates_scaled = np.load('in/xtr/rt_20_odt_rates_30_scaled.npy')
    odt_stop_ids = np.load('in/xtr/rt_20_odt_stops.npy')
    odt_stop_ids = list(odt_stop_ids)
    odt_stop_ids = [str(int(s)) for s in odt_stop_ids]
    dep_delay_dist_out = load('in/xtr/dep_delay_dist_out.pkl')
    return stops_out, link_times_info, trips_out_info, odt_rates_scaled, odt_stop_ids, sched_arrivals, odt_rates_old, dep_delay_dist_out


def get_params_inbound():
    trip_times1_params = load('in/xtr/trip_time1_params.pkl')
    trip_times2_params = load('in/xtr/trip_time2_params.pkl')
    trips1_out_info = load('in/xtr/trips1_info_inbound.pkl')
    trips2_out_info = load('in/xtr/trips2_info_inbound.pkl')
    deadhead_times_params = load('in/xtr/deadhead_times_params.pkl')
    sched_arrs = load('in/xtr/scheduled_arrivals_inbound.pkl')
    dep_delay1_dist_in = load('in/xtr/dep_delay1_dist_in.pkl')
    dep_delay2_dist_in = load('in/xtr/dep_delay2_dist_in.pkl')
    trip_t1_dist_in = load('in/xtr/trip_t1_dist_in.pkl')
    trip_t2_dist_in = load('in/xtr/trip_t2_dist_in.pkl')
    return trip_times1_params, trip_times2_params, trips1_out_info, trips2_out_info, deadhead_times_params, sched_arrs, dep_delay1_dist_in, dep_delay2_dist_in, trip_t1_dist_in, trip_t2_dist_in


# extract_params(outbound_route_params=True)
# analyze_inbound(path_avl, START_TIME_SEC, END_TIME_SEC, DELAY_INTERVAL_LENGTH_MINS)
STOPS_OUTBOUND, LINK_TIMES_INFO, TRIPS_OUT_INFO, SCALED_ODT_RATES, ODT_STOP_IDS, SCHED_ARRS_OUT, ODT_RATES_OLD, DEP_DELAY_DIST_OUT = get_params_outbound()
TRIP_TIMES1_PARAMS, TRIP_TIMES2_PARAMS, TRIPS1_IN_INFO, TRIPS2_IN_INFO, DEADHEAD_TIME_PARAMS, SCHED_ARRS_IN, DEP_DELAY1_DIST_IN, DEP_DELAY2_DIST_IN, TRIP_T1_DIST_IN, TRIP_T2_DIST_IN = get_params_inbound()


trip_t1_dist_in_samp = [[] for _ in range(len(TRIP_T1_DIST_IN))]
trip_t2_dist_in_samp = [[] for _ in range(len(TRIP_T2_DIST_IN))]
n_samples = 30
for i in range(len(TRIP_T1_DIST_IN)):
    sample_percentiles = np.random.uniform(low=0.0, high=100.0, size=n_samples)
    if TRIP_T1_DIST_IN[i]:
        trip_t1_dist_in_samp[i] = list(np.percentile(TRIP_T1_DIST_IN[i], sample_percentiles))
    if TRIP_T2_DIST_IN[i]:
        trip_t2_dist_in_samp[i] = list(np.percentile(TRIP_T2_DIST_IN[i], sample_percentiles))
fig, axs = plt.subplots(nrows=2, ncols=2, sharey='col', sharex='col')
for i in range(2, 4):
    arr1 = np.array(TRIP_T1_DIST_IN[i])/60
    arr2 = np.array(trip_t1_dist_in_samp[i])/60
    axs[i - 2, 0].hist([arr1, arr2], density=True,
                       label=['historical', 'sampled'])
    arr1 = np.array(TRIP_T2_DIST_IN[i])/60
    arr2 = np.array(trip_t2_dist_in_samp[i])/60
    axs[i - 2, 1].hist([arr1, arr2], density=True,
                       label=['historical', 'sampled'])
axs[0, 0].set_title('long')
axs[0, 1].set_title('short')
plt.legend()
plt.tight_layout()
plt.savefig('out/compare/validate/trip_t_inbound.png')
plt.close()


LINK_TIMES_MEAN, LINK_TIMES_EXTREMES, LINK_TIMES_PARAMS = LINK_TIMES_INFO
SCALED_ARR_RATES = np.sum(SCALED_ODT_RATES, axis=-1)

TRIP_IDS_OUT, SCHED_DEP_OUT, BLOCK_IDS_OUT = [], [], []
for item in TRIPS_OUT_INFO:
    TRIP_IDS_OUT.append(item[0]), SCHED_DEP_OUT.append(item[1]), BLOCK_IDS_OUT.append(item[2])
trips_out = [(x, y, str(timedelta(seconds=y)), z, 0, w, v) for x, y, z, w, v in TRIPS_OUT_INFO]
trips_in1 = [(x, y, str(timedelta(seconds=y)), z, 1, [], []) for x, y, z in TRIPS1_IN_INFO]
trips_in2 = [(x, y, str(timedelta(seconds=y)), z, 2, [], []) for x, y, z in TRIPS2_IN_INFO]

trips_df = pd.DataFrame(trips_out + trips_in1 + trips_in2,
                        columns=['trip_id', 'schd_sec', 'schd_time', 'block_id', 'route_type', 'schedule', 'stops'])
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
# trips_df.to_csv('in/vis/block_info.csv', index=False)
block_ids = trips_df['block_id'].unique().tolist()
BLOCK_TRIPS_INFO = []
BLOCK_DICT = {}
# avl_df = pd.read_csv('in/raw/rt20_avl.csv')
for b in block_ids:
    block_df = trips_df[trips_df['block_id'] == b]
    trip_ids = block_df['trip_id'].tolist()
    sched_deps = block_df['schd_sec'].tolist()
    lst_stops = block_df['stops'].tolist()
    lst_schedule = block_df['schedule'].tolist()
    BLOCK_DICT[b] = trip_ids
    route_types = block_df['route_type'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, sched_deps, route_types, lst_stops, lst_schedule))))
# print(BLOCK_TRIPS_INFO)

# layover_t = {'2-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#              '1-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#              '0-1': [[] for _ in range(TRIP_TIME_NR_INTERVALS)]}
# after_layover_t = {'2-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                    '1-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                    '0-1': [[] for _ in range(TRIP_TIME_NR_INTERVALS)]}
# sched_layover_t = {'2-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                    '1-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                    '0-1': [[] for _ in range(TRIP_TIME_NR_INTERVALS)]}
# late_layover_t = {'2-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                   '1-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#                   '0-1': [[] for _ in range(TRIP_TIME_NR_INTERVALS)]}
# delay = {'2-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#          '1-0': [[] for _ in range(TRIP_TIME_NR_INTERVALS)],
#          '0-1': [[] for _ in range(TRIP_TIME_NR_INTERVALS)]}
#     terminal1_id = 386
#     prev_terminal1_id = 14800
#     after_terminal1_id = 388
#
#     terminal2_id = 8613
#     prev_terminal2_id = 3954
#     after_terminal2_id = 6360
#
#     for i in range(len(trip_ids) - 1):
#         rt_from = int(route_types[i])
#         rt_to = int(route_types[i + 1])
#         trip_from = trip_ids[i]
#         trip_to = trip_ids[i + 1]
#         df_trip_from = avl_df[avl_df['trip_id'] == trip_from]
#         df_trip_to = avl_df[avl_df['trip_id'] == trip_to]
#         rt_from_to = str(rt_from) + '-' + str(rt_to)
#         interval_idx = get_interval(sched_deps[i + 1], TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
#         for d in DATES:
#             df_trip_from_temp = df_trip_from[df_trip_from['avl_arr_time'].astype(str).str[:10] == d]
#             df_trip_to_temp = df_trip_to[df_trip_to['avl_arr_time'].astype(str).str[:10] == d]
#             if rt_from == 1 or rt_from == 2:
#                 terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == terminal1_id]['avl_arr_sec'].tolist()
#                 prev_terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == prev_terminal1_id][
#                     'avl_arr_sec'].tolist()
#                 sched_terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == terminal1_id][
#                     'schd_sec'].tolist()
#
#                 terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == terminal1_id]['avl_dep_sec'].tolist()
#                 after_terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == after_terminal1_id][
#                     'avl_dep_sec'].tolist()
#                 sched_terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == terminal1_id][
#                     'schd_sec'].tolist()
#             else:
#                 assert rt_from == 0 and rt_to == 1
#                 terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == terminal2_id]['avl_arr_sec'].tolist()
#                 prev_terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == prev_terminal2_id][
#                     'avl_arr_sec'].tolist()
#                 sched_terminal_arr = df_trip_from_temp[df_trip_from_temp['stop_id'] == terminal2_id][
#                     'schd_sec'].tolist()
#
#                 terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == terminal2_id]['avl_dep_sec'].tolist()
#                 after_terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == after_terminal2_id][
#                     'avl_dep_sec'].tolist()
#                 sched_terminal_dep = df_trip_to_temp[df_trip_to_temp['stop_id'] == terminal2_id][
#                     'schd_sec'].tolist()
#             if terminal_arr and terminal_dep:
#                 layover_t[rt_from_to][interval_idx].append(terminal_dep[0] - terminal_arr[0])
#                 delay[rt_from_to][interval_idx].append(terminal_dep[0]%86400 - sched_terminal_dep[0])
#             if prev_terminal_arr and after_terminal_dep:
#                 after_layover_t[rt_from_to][interval_idx].append(after_terminal_dep[0] - prev_terminal_arr[0])
#             if sched_terminal_arr and sched_terminal_dep:
#                 sched_layover_t[rt_from_to][interval_idx].append(sched_terminal_dep[0] - sched_terminal_arr[0])
#             if sched_terminal_dep and terminal_arr and terminal_dep:
#                 if (terminal_dep[0]%86400) - sched_terminal_dep[0] > 50:
#                     late_layover_t[rt_from_to][interval_idx].append(terminal_dep[0] - terminal_arr[0])
#
# # print(layover_t['2-0'])
# print(sched_layover_t)
# # print(late_layover_t['2-0'])
# mean_layover = {'1-0': [], '2-0': [], '0-1': []}
# mean_late_layover = {'1-0': [], '2-0': [], '0-1': []}
# for terminal_combo in layover_t:
#     fig, axs = plt.subplots(nrows=3, ncols=2, sharey='all', sharex='all')
#     # late_layover_combo = late_layover_t[terminal_combo]
#     layover_combo = layover_t[terminal_combo]
#     for i in range(2, TRIP_TIME_NR_INTERVALS-2):
#         # if late_layover_combo[i]:
#         #     mean_late_layover[terminal_combo].append([np.around(np.mean(late_layover_combo[i])),
#         #                                               np.around(np.std(late_layover_combo[i]))])
#         # else:
#         #     mean_late_layover[terminal_combo].append(np.nan)
#         if layover_combo[i]:
#             delay_arr = np.array(delay[terminal_combo][i])
#             layover_arr = np.array(layover_combo[i])
#             axs.flat[i-2].scatter(delay_arr[(delay_arr>0)& (delay_arr<400)&(layover_arr<400)], layover_arr[(delay_arr>0)& (delay_arr<400)&(layover_arr<400)])
#             axs.flat[i-2].grid()
#             mean_layover[terminal_combo].append([np.around(np.mean(layover_combo[i])),
#                                                  np.around(np.std(layover_combo[i]))])
#         else:
#             mean_layover[terminal_combo].append(np.nan)
#     axs.flat[-1].set_xlabel('dep delay (sec)')
#     axs.flat[0].set_ylabel('layover time (sec)')
#     plt.tight_layout()
#     plt.savefig('in/vis/layover_' + terminal_combo + '.png')

PAX_INIT_TIME = [0]
for s0, s1 in zip(STOPS_OUTBOUND, STOPS_OUTBOUND[1:]):
    ltimes = np.array(LINK_TIMES_MEAN[s0 + '-' + s1])
    ltime = ltimes[np.isfinite(ltimes)][0]
    PAX_INIT_TIME.append(ltime)
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_OUT[0] - ((SCHED_DEP_OUT[1] - SCHED_DEP_OUT[0]) / 2)

# trip id focused for results
ordered_trips_arr = np.array([TRIP_IDS_OUT])
scheduled_deps_arr = np.array([SCHED_DEP_OUT])
FOCUS_TRIPS = ordered_trips_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
FOCUS_TRIPS_SCHED = scheduled_deps_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
focus_trips_hw = [i - j for i, j in zip(FOCUS_TRIPS_SCHED[1:], FOCUS_TRIPS_SCHED[:-1])]
FOCUS_TRIPS_MEAN_HW = np.mean(focus_trips_hw)
FOCUS_TRIPS_HW_CV = round(np.std(focus_trips_hw) / np.mean(focus_trips_hw), 2)
LAST_FOCUS_TRIP = FOCUS_TRIPS[-1]
LAST_FOCUS_TRIP_BLOCK = trips_df[trips_df['trip_id'] == LAST_FOCUS_TRIP]['block_id'].tolist()[0]
LAST_FOCUS_TRIP_BLOCK_IDX = block_ids.index(LAST_FOCUS_TRIP_BLOCK)
FOCUS_TRIP_IDS_OUT_LONG = [ti[0] for ti in TRIPS1_IN_INFO if
                           (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_IDS_OUT_SHORT = [ti[0] for ti in TRIPS2_IN_INFO if
                            (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_DEP_T_OUT_LONG = [ti[1] for ti in TRIPS1_IN_INFO if
                             (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_DEP_T_OUT_SHORT = [ti[1] for ti in TRIPS2_IN_INFO if
                              (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]

# trip id focused for control
NO_CONTROL_TRIP_IDS = TRIP_IDS_OUT[:9] + TRIP_IDS_OUT[-11:]
NO_CONTROL_SCHED = SCHED_DEP_OUT[:9] + SCHED_DEP_OUT[-11:]
CONTROL_TRIP_IDS = TRIP_IDS_OUT[9:-11]
CONTROL_SCHEDULE = SCHED_DEP_OUT[9:-11]
CONTROL_HW = [t1 - t0 for t1, t0 in zip(CONTROL_SCHEDULE[1:], CONTROL_SCHEDULE[:-1])]
CONTROL_MEAN_HW = sum(CONTROL_HW) / len(CONTROL_HW)

BASE_HOLDING_TIME = 25
MIN_HW_THRESHOLD = 0.4
LIMIT_HOLDING = int(MIN_HW_THRESHOLD * CONTROL_MEAN_HW - MIN_HW_THRESHOLD * CONTROL_MEAN_HW % BASE_HOLDING_TIME)
N_ACTIONS_RL = int(LIMIT_HOLDING / BASE_HOLDING_TIME) + 2

# FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
UNIFORM_INTERVAL = 1
SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}
SINGLE_LINK_TIMES_PARAMS = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_PARAMS.items()}
SINGLE_LINK_TIMES_EXTREMES = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_EXTREMES.items()}
