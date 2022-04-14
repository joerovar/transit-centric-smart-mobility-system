from pre_process import *
from file_paths import *
from constants import *
from post_process import save, load


def extract_params(inbound_route_params=False, outbound_route_params=False, demand=False, validation=False):
    if inbound_route_params:
        stops, trips_in_info, link_times_info, sched_arrs_in = get_route(path_stop_times,
                                                                         START_TIME_SEC, END_TIME_SEC,
                                                                         TIME_NR_INTERVALS,
                                                                         TIME_START_INTERVAL, TIME_INTERVAL_LENGTH_MINS,
                                                                         DATES, TRIP_WITH_FULL_STOP_PATTERN,
                                                                         path_avl)
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_info)
        save('in/xtr/trips_inbound_info.pkl', trips_in_info)
        save('in/xtr/scheduled_arrivals_inbound.pkl', sched_arrs_in)
        stop_df = pd.read_csv('in/raw/gtfs/stops.txt')
        stop_df = stop_df[stop_df['stop_id'].isin([int(s) for s in stops])]

        stop_seq_dict = {'stop_id': [int(s) for s in stops], 'stop_seq': [i for i in range(1, len(stops) + 1)]}
        stop_seq_df = pd.DataFrame(stop_seq_dict)
        stop_df = pd.merge(stop_df, stop_seq_df, on='stop_id')
        stop_df = stop_df.sort_values(by='stop_seq')
        stop_df = stop_df[['stop_seq', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        stop_df.to_csv('in/raw/rt20_in_stops.txt', index=False)

    if outbound_route_params:
        trips1_info_out, trips2_info_out, sched_arrs_out, trip_times1_params, trip_times2_params, \
        deadhead_time_params = get_outbound_travel_time(
            path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
            TRIP_TIME_INTERVAL_LENGTH_MINS)
        save('in/xtr/trips1_info_outbound.pkl', trips1_info_out)
        save('in/xtr/trips2_info_outbound.pkl', trips2_info_out)
        save('in/xtr/scheduled_arrivals_outbound.pkl', sched_arrs_out)
        save('in/xtr/trip_time1_params.pkl', trip_times1_params)
        save('in/xtr/trip_time2_params.pkl', trip_times2_params)
        save('in/xtr/deadhead_times_params.pkl', deadhead_time_params)
    if demand:
        stops = load(path_route_stops)
        # arrival rates will be in pax/min
        arrival_rates, alight_rates, odt = get_demand(path_od, path_stop_times, stops, INPUT_DEM_START_INTERVAL,
                                                      INPUT_DEM_END_INTERVAL, DEM_START_INTERVAL, DEM_END_INTERVAL,
                                                      DEM_PROPORTION_INTERVALS, DEM_INTERVAL_LENGTH_MINS, DATES)
        save(path_odt_xtr, odt)
    if validation:
        stops = load(path_route_stops)
        trips_inbound_info = load('in/xtr/trips_inbound_info.pkl')
        scheduled_dep_in, ordered_trips_in = [], []
        for t in trips_inbound_info:
            scheduled_dep_in.append(t[1]), ordered_trips_in.append(t[0])
        ordered_trips_in = np.array(ordered_trips_in)
        schedule_arr = np.array(scheduled_dep_in)
        focus_trips = ordered_trips_in[
            (schedule_arr <= FOCUS_END_TIME_SEC) & (schedule_arr >= FOCUS_START_TIME_SEC)].tolist()
        trip_times, headway_in, headway_in_cv = get_trip_times(path_avl, focus_trips, DATES, stops)
        save('in/xtr/trip_times_inbound.pkl', trip_times)
        save('in/xtr/departure_headway_inbound.pkl', headway_in)
        save('in/xtr/cv_headway_inbound.pkl', headway_in_cv)
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


def get_params_inbound():
    stops = load(path_route_stops)
    link_times_info = load(path_link_times_mean)
    trips_in_info = load('in/xtr/trips_inbound_info.pkl')
    odt = load(path_odt_xtr)
    sched_arrivals = load('in/xtr/scheduled_arrivals_inbound.pkl')
    trip_times = load('in/xtr/trip_times_inbound.pkl')
    return stops, link_times_info, trips_in_info, odt, sched_arrivals, trip_times


def get_params_outbound():
    trip_times1_params = load('in/xtr/trip_time1_params.pkl')
    trip_times2_params = load('in/xtr/trip_time2_params.pkl')
    trips1_out_info = load('in/xtr/trips1_info_outbound.pkl')
    trips2_out_info = load('in/xtr/trips2_info_outbound.pkl')
    deadhead_times_params = load('in/xtr/deadhead_times_params.pkl')
    sched_arrs = load('in/xtr/scheduled_arrivals_outbound.pkl')
    return trip_times1_params, trip_times2_params, trips1_out_info, trips2_out_info, deadhead_times_params, sched_arrs


# extract_params(inbound_route_params=True)

STOPS, LINK_TIMES_INFO, TRIPS_IN_INFO, ODT, SCHED_ARRS_IN, TRIP_TIMES_INPUT = get_params_inbound()
TRIP_TIMES1_PARAMS, TRIP_TIMES2_PARAMS, TRIPS1_INFO_OUT, TRIPS2_INFO_OUT, DEADHEAD_TIME_PARAMS, SCHED_ARRS_OUT = get_params_outbound()
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
ARR_RATES = np.nansum(ODT, axis=-1)
PAX_INIT_TIME = [0] + [LINK_TIMES_MEAN[s0 + '-' + s1][0] for s0, s1 in zip(STOPS, STOPS[1:])]
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
