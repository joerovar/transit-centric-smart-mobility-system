from pre_process import *
from file_paths import *
from constants import *
from post_process import save, load


def extract_params(outbound_route_params=False, inbound_route_params=False, demand=False, validation=False):
    if outbound_route_params:
        stops, trips_in_info, link_times_info, sched_arrs_in = get_route(path_stop_times,
                                                                         START_TIME_SEC, END_TIME_SEC,
                                                                         TIME_NR_INTERVALS,
                                                                         TIME_START_INTERVAL, TIME_INTERVAL_LENGTH_MINS,
                                                                         DATES, TRIP_WITH_FULL_STOP_PATTERN,
                                                                         path_avl)
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_info)
        save('in/xtr/trips_outbound_info.pkl', trips_in_info)
        save('in/xtr/scheduled_arrivals_outbound.pkl', sched_arrs_in)
        stop_df = pd.read_csv('in/raw/gtfs/stops.txt')
        stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in stops])]

        stop_seq_dict = {'stop_id': [int(s) for s in stops], 'stop_seq': [i for i in range(1, len(stops) + 1)]}
        stop_seq_df = pd.DataFrame(stop_seq_dict)
        stop_df = pd.merge(stop_df, stop_seq_df, on='stop_id')
        stop_df = stop_df.sort_values(by='stop_seq')
        stop_df = stop_df[['stop_seq', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        stop_df.to_csv('in/raw/rt20_in_stops.txt', index=False)

    if inbound_route_params:
        trips1_info_out, trips2_info_out, sched_arrs_out, trip_times1_params, trip_times2_params, \
        deadhead_time_params = get_inbound_travel_time(
            path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
            TRIP_TIME_INTERVAL_LENGTH_MINS)
        save('in/xtr/trips1_info_inbound.pkl', trips1_info_out)
        save('in/xtr/trips2_info_inbound.pkl', trips2_info_out)
        save('in/xtr/scheduled_arrivals_inbound.pkl', sched_arrs_out)
        save('in/xtr/trip_time1_params.pkl', trip_times1_params)
        save('in/xtr/trip_time2_params.pkl', trip_times2_params)
        save('in/xtr/deadhead_times_params.pkl', deadhead_time_params)

    # if demand:
    #     stops = load(path_route_stops)
    #     # arrival rates will be in pax/min
    #     arrival_rates, alight_rates, odt_rates = get_demand(path_od, path_stop_times, stops, INPUT_DEM_START_INTERVAL,
    #                                                         INPUT_DEM_END_INTERVAL, DEM_START_INTERVAL,
    #                                                         DEM_END_INTERVAL,
    #                                                         DEM_PROPORTION_INTERVALS, DEM_INTERVAL_LENGTH_MINS, DATES)
    #     save(path_odt_rates_xtr, odt_rates)

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
        trip_times, headway_out, headway_out_cv = get_trip_times(path_avl, focus_trips, DATES, stops)
        save('in/xtr/trip_times_outbound.pkl', trip_times)
        save('in/xtr/departure_headway_outbound.pkl', headway_out)
        save('in/xtr/cv_headway_outbound.pkl', headway_out_cv)
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
    return


def get_params_outbound():
    stops = load(path_route_stops)
    link_times_info = load(path_link_times_mean)
    trips_in_info = load('in/xtr/trips_outbound_info.pkl')
    odt_rates_old = load(path_odt_rates_xtr)
    sched_arrivals = load('in/xtr/scheduled_arrivals_outbound.pkl')
    trip_times = load('in/xtr/trip_times_outbound.pkl')
    odt_rates = np.load('in/xtr/rt_20_odt_rates_30.npy')
    odt_stop_ids = np.load('in/xtr/rt_20_odt_stops.npy')
    odt_stop_ids = list(odt_stop_ids)
    odt_stop_ids = [str(int(s)) for s in odt_stop_ids]

    return stops, link_times_info, trips_in_info, odt_rates, odt_stop_ids, sched_arrivals, trip_times, odt_rates_old


def get_params_inbound():
    trip_times1_params = load('in/xtr/trip_time1_params.pkl')
    trip_times2_params = load('in/xtr/trip_time2_params.pkl')
    trips1_out_info = load('in/xtr/trips1_info_inbound.pkl')
    trips2_out_info = load('in/xtr/trips2_info_inbound.pkl')
    deadhead_times_params = load('in/xtr/deadhead_times_params.pkl')
    sched_arrs = load('in/xtr/scheduled_arrivals_inbound.pkl')
    return trip_times1_params, trip_times2_params, trips1_out_info, trips2_out_info, deadhead_times_params, sched_arrs


# extract_params(outbound_route_params=True, inbound_route_params=True, validation=True)

STOPS_OUTBOUND, LINK_TIMES_INFO, TRIPS_IN_INFO, ODT_RATES, ODT_STOP_IDS ,SCHED_ARRS_IN, TRIP_TIMES_INPUT, ODT_RATES_OLD = get_params_outbound()
TRIP_TIMES1_PARAMS, TRIP_TIMES2_PARAMS, TRIPS1_INFO_OUT, TRIPS2_INFO_OUT, DEADHEAD_TIME_PARAMS, SCHED_ARRS_OUT = get_params_inbound()
TRIP_IDS_OUT = [ti[0] for ti in TRIPS1_INFO_OUT]
TRIP_IDS_OUT += [ti[0] for ti in TRIPS2_INFO_OUT]
LINK_TIMES_MEAN, LINK_TIMES_EXTREMES, LINK_TIMES_PARAMS = LINK_TIMES_INFO

TRIP_IDS_IN, SCHED_DEP_IN, BLOCK_IDS_IN = [], [], []
for item in TRIPS_IN_INFO:
    TRIP_IDS_IN.append(item[0]), SCHED_DEP_IN.append(item[1]), BLOCK_IDS_IN.append(item[2])
trips_in = [(x, y, str(timedelta(seconds=y)), z, 0) for x, y, z in TRIPS_IN_INFO]
trips_out1 = [(x, y, str(timedelta(seconds=y)), z, 1) for x, y, z in TRIPS1_INFO_OUT]
trips_out2 = [(x, y, str(timedelta(seconds=y)), z, 2) for x, y, z in TRIPS2_INFO_OUT]
trips_df = pd.DataFrame(trips_in + trips_out1 + trips_out2,
                        columns=['trip_id', 'schd_sec', 'schd_time', 'block_id', 'route_type'])
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
# trips_df.to_csv('in/vis/block_info.csv', index=False)
block_ids = trips_df['block_id'].unique().tolist()
BLOCK_TRIPS_INFO = []
BLOCK_DICT = {}
for b in block_ids:
    block_df = trips_df[trips_df['block_id'] == b]
    trip_ids = block_df['trip_id'].tolist()
    sched_deps = block_df['schd_sec'].tolist()
    BLOCK_DICT[b] = trip_ids
    trip_routes = block_df['route_type'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, sched_deps, trip_routes))))

# demand
ARR_RATES = np.nansum(ODT_RATES, axis=-1)
idx_outbound = [ODT_STOP_IDS.index(s) for s in STOPS_OUTBOUND]
outbound_arr_rates = ARR_RATES[:, idx_outbound]
outbound_arr_tot = np.sum(outbound_arr_rates, axis=-1)

apc_on_counts = load('in/xtr/apc_on_counts.pkl')
outbound_on_counts = apc_on_counts[:, idx_outbound]
outbound_on_tot_count = np.nansum(outbound_on_counts, axis=-1)

x = np.arange(outbound_on_tot_count.shape[0])
plt.plot(x, outbound_arr_tot, label='predicted')
plt.plot(x, outbound_on_tot_count, label='apc')
plt.xticks(np.arange(0, outbound_on_tot_count.shape[0], 2), np.arange(int(outbound_on_tot_count.shape[0]/2)))
plt.xlabel('hour of day')
plt.ylabel('arrival rate (1/h)')
plt.legend()
# plt.show()
plt.close()

SHIFTED_ODT = np.concatenate((ODT_RATES[-6:], ODT_RATES[:-6]), axis=0)
ARR_RATES_S = np.nansum(SHIFTED_ODT, axis=-1)
out_arr_rates_s = ARR_RATES_S[:, idx_outbound]
outbound_arr_tot_s = np.sum(out_arr_rates_s, axis=-1)

x = np.arange(outbound_on_tot_count.shape[0])
plt.plot(x, outbound_arr_tot_s, label='predicted')
plt.plot(x, outbound_on_tot_count, label='apc')
plt.xticks(np.arange(0, outbound_on_tot_count.shape[0], 2), np.arange(int(outbound_on_tot_count.shape[0]/2)))
plt.xlabel('hour of day')
plt.ylabel('arrival rate (1/h)')
plt.yticks(np.arange(0, 1200, 200))
plt.legend()
# plt.show()
plt.close()

SCALED_ODT = np.load('in/xtr/rt_20_odt_rates_30_scaled.npy')
SCALED_ARR_RATES = np.sum(SCALED_ODT, axis=-1)
scaled_out_arr_rates = SCALED_ARR_RATES[:, idx_outbound]
scaled_out_tot = np.sum(scaled_out_arr_rates, axis=-1)

ARR_RATES_OLD = np.nansum(ODT_RATES_OLD, axis=-1)
old_out_tot = np.nansum(ARR_RATES_OLD, axis=-1)

x = np.arange(outbound_on_tot_count.shape[0])
plt.plot(np.arange(10, 20, 2), old_out_tot, label='previous odt')
plt.plot(x, scaled_out_tot, label='odt scaled')
plt.plot(x, outbound_arr_tot_s, label='odt')
plt.plot(x, outbound_on_tot_count, label='apc')
plt.xticks(np.arange(0, outbound_on_tot_count.shape[0], 2), np.arange(int(outbound_on_tot_count.shape[0]/2)))
plt.xlabel('hour of day')
plt.ylabel('arrival rate (1/h)')
plt.yticks(np.arange(0, 1200, 200))
plt.legend()
# plt.show()
plt.close()

print(scaled_out_tot[ODT_START_INTERVAL:ODT_END_INTERVAL])
print(old_out_tot)

PAX_INIT_TIME = [0]
for s0, s1 in zip(STOPS_OUTBOUND, STOPS_OUTBOUND[1:]):
    ltimes = np.array(LINK_TIMES_MEAN[s0 + '-' + s1])
    ltime = ltimes[np.isfinite(ltimes)][0]
    PAX_INIT_TIME.append(ltime)
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_IN[0] - ((SCHED_DEP_IN[1] - SCHED_DEP_IN[0]) / 2)

# trip id focused for results
ordered_trips_arr = np.array([TRIP_IDS_IN])
scheduled_deps_arr = np.array([SCHED_DEP_IN])
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
FOCUS_TRIP_IDS_OUT_LONG = [ti[0] for ti in TRIPS1_INFO_OUT if
                           (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_IDS_OUT_SHORT = [ti[0] for ti in TRIPS2_INFO_OUT if
                            (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_DEP_T_OUT_LONG = [ti[1] for ti in TRIPS1_INFO_OUT if
                             (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]
FOCUS_TRIP_DEP_T_OUT_SHORT = [ti[1] for ti in TRIPS2_INFO_OUT if
                              (ti[1] > START_TIME_SEC + 3600) and (ti[1] < END_TIME_SEC - 6400)]

# trip id focused for control
NO_CONTROL_TRIP_IDS = TRIP_IDS_IN[:9] + TRIP_IDS_IN[-11:]
NO_CONTROL_SCHED = SCHED_DEP_IN[:9] + SCHED_DEP_IN[-11:]
CONTROL_TRIP_IDS = TRIP_IDS_IN[9:-11]
CONTROL_SCHEDULE = SCHED_DEP_IN[9:-11]
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
