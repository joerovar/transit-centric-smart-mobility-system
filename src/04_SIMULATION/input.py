from datetime import timedelta
from pre_process import *
from post_process import *
from file_paths import *
from constants import *


def extract_params(route_params=False, demand=False, validation=False):
    if route_params:
        stops, ordered_trips, link_times_true, sched_deps_in, sched_arrs_in, bus_ids_in = get_route(path_stop_times,
                                                                                                    START_TIME_SEC,
                                                                                                    END_TIME_SEC,
                                                                                                    TIME_NR_INTERVALS,
                                                                                                    TIME_START_INTERVAL,
                                                                                                    TIME_INTERVAL_LENGTH_MINS,
                                                                                                    DATES,
                                                                                                    TRIP_WITH_FULL_STOP_PATTERN)

        link_times_mean_true, link_times_sd_true, nr_time_dpoints_true = link_times_true
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_mean_true)
        save(path_ordered_trips, ordered_trips)
        save(path_departure_times_xtr, sched_deps_in)
        save('in/xtr/rt_20-2019-09/scheduled_arrivals_inbound.pkl', sched_arrs_in)
        save('in/xtr/rt_20-2019-09/bus_ids_inbound.pkl', bus_ids_in)
        trips1_info_out, trips2_info_out, sched_arrs_out, trip_times1_params, trip_times2_params, \
        deadhead_time_params = get_outbound_travel_time(
            path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
            TRIP_TIME_INTERVAL_LENGTH_MINS)
        save('in/xtr/rt_20-2019-09/trips1_info_outbound.pkl', trips1_info_out)
        save('in/xtr/rt_20-2019-09/trips2_info_outbound.pkl', trips2_info_out)
        save('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl', sched_arrs_out)
        save('in/xtr/rt_20-2019-09/trip_time1_params.pkl', trip_times1_params)
        save('in/xtr/rt_20-2019-09/trip_time2_params.pkl', trip_times2_params)
        save('in/xtr/rt_20-2019-09/deadhead_times_params.pkl', deadhead_time_params)
    if demand:
        stops = load(path_route_stops)
        # arrival rates will be in pax/min
        arrival_rates, alight_rates, odt = get_demand(path_od, path_stop_times, stops, INPUT_DEM_START_INTERVAL,
                                                      INPUT_DEM_END_INTERVAL, DEM_START_INTERVAL, DEM_END_INTERVAL,
                                                      DEM_PROPORTION_INTERVALS, DEM_INTERVAL_LENGTH_MINS, DATES)
        save(path_odt_xtr, odt)
    if validation:
        schedule = load(path_departure_times_xtr)
        trip_ids = load(path_ordered_trips)
        stops = load(path_route_stops)
        ordered_trips = load(path_ordered_trips)
        schedule_arr = np.array(schedule)
        trip_ids_arr = np.array(trip_ids)
        focus_trips = trip_ids_arr[
            (schedule_arr <= FOCUS_END_TIME_SEC) & (schedule_arr >= FOCUS_START_TIME_SEC)].tolist()
        trip_times, departure_headway_in = get_trip_times(path_stop_times, focus_trips, DATES, START_TIME_SEC,
                                                          END_TIME_SEC)
        save('in/xtr/rt_20-2019-09/trip_times_inbound.pkl', trip_times)

        dwell_times_mean, dwell_times_std, dwell_times_tot = get_dwell_times(path_stop_times, focus_trips, stops, DATES)
        save('in/xtr/rt_20-2019-09/dwell_times_mean.pkl', dwell_times_mean)
        save('in/xtr/rt_20-2019-09/dwell_times_std.pkl', dwell_times_std)
        save('in/xtr/rt_20-2019-09/dwell_times_tot.pkl', dwell_times_tot)
        save('in/xtr/rt_20-2019-09/departure_headway_inbound.pkl', departure_headway_in)
        write_inbound_trajectories(path_stop_times, ordered_trips)
        load_profile = get_load_profile(path_stop_times, focus_trips, stops)
        save('in/xtr/rt_20-2019-09/load_profile.pkl', load_profile)
    return


def get_params_inbound():
    stops = load(path_route_stops)
    link_times_mean = load(path_link_times_mean)
    ordered_trips = load(path_ordered_trips)
    scheduled_departures = load(path_departure_times_xtr)
    odt = load(path_odt_xtr)
    sched_arrivals = load('in/xtr/rt_20-2019-09/scheduled_arrivals_inbound.pkl')
    trip_times = load('in/xtr/rt_20-2019-09/trip_times_inbound.pkl')
    bus_ids = load('in/xtr/rt_20-2019-09/bus_ids_inbound.pkl')
    return stops, link_times_mean, ordered_trips, scheduled_departures, odt, sched_arrivals, trip_times, bus_ids


def get_params_outbound():
    trip_times1_params = load('in/xtr/rt_20-2019-09/trip_time1_params.pkl')
    trip_times2_params = load('in/xtr/rt_20-2019-09/trip_time2_params.pkl')
    trips1_out_info = load('in/xtr/rt_20-2019-09/trips1_info_outbound.pkl')
    trips2_out_info = load('in/xtr/rt_20-2019-09/trips2_info_outbound.pkl')
    deadhead_times_params = load('in/xtr/rt_20-2019-09/deadhead_times_params.pkl')
    sched_arrs = load('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl')
    return trip_times1_params, trip_times2_params, trips1_out_info, trips2_out_info, deadhead_times_params, sched_arrs


# extract_params(route_params=True, validation=True)

STOPS, LINK_TIMES_MEAN, TRIP_IDS_IN, SCHED_DEP_IN, ODT, SCHED_ARRS_IN, TRIP_TIMES_INPUT, BUS_IDS_IN = get_params_inbound()
TRIP_TIMES1_PARAMS, TRIP_TIMES2_PARAMS, TRIPS1_INFO_OUT, TRIPS2_INFO_OUT, DEADHEAD_TIME_PARAMS, SCHED_ARRS_OUT = get_params_outbound()

trips_in = [(x, y, str(timedelta(seconds=y)), z, 0) for x, y, z in zip(TRIP_IDS_IN, SCHED_DEP_IN, BUS_IDS_IN)]
trips_out1 = [(x, y, str(timedelta(seconds=y)), z, 1) for x, y, z in TRIPS1_INFO_OUT]
trips_out2 = [(x, y, str(timedelta(seconds=y)), z, 2) for x, y, z in TRIPS2_INFO_OUT]

trips_df = pd.DataFrame(trips_in + trips_out1 + trips_out2, columns=['trip_id', 'schd_sec', 'schd_time', 'block_id', 'route_type'])
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
block_ids = trips_df['block_id'].unique().tolist()
BLOCK_TRIPS_INFO = []
for b in block_ids:
    block_df = trips_df[trips_df['block_id'] == b]
    trip_ids = block_df['trip_id'].tolist()
    sched_deps = block_df['schd_sec'].tolist()
    trip_routes = block_df['route_type'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, sched_deps, trip_routes))))

warm_up_odt = ODT[4]
for i in range(4):
    ODT = np.insert(ODT, 0, warm_up_odt, axis=0)
cool_down_odt = ODT[4]
ODT = np.insert(ODT, -1, cool_down_odt, axis=0)
# print(ODT.shape)
# SCHEDULED_DEPARTURES = UNIFORM_SCHEDULED_DEPARTURES.copy()
PAX_INIT_TIME = [0] + [LINK_TIMES_MEAN[s0 + '-' + s1][0] for s0, s1 in zip(STOPS, STOPS[1:])]
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_IN[0] - (SCHED_DEP_IN[1] - SCHED_DEP_IN[0])
# print([str(timedelta(seconds=i)) for i in SCHEDULED_DEPARTURES])
ordered_trips_arr = np.array([TRIP_IDS_IN])
scheduled_deps_arr = np.array([SCHED_DEP_IN])
FOCUS_TRIPS = ordered_trips_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
LAST_FOCUS_TRIP = FOCUS_TRIPS[-1]
LAST_FOCUS_TRIP_BLOCK = trips_df[trips_df['trip_id'] == LAST_FOCUS_TRIP]['block_id'].tolist()[0]
LAST_FOCUS_TRIP_BLOCK_IDX = block_ids.index(LAST_FOCUS_TRIP_BLOCK)
# FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
UNIFORM_INTERVAL = 1
SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}
