from datetime import timedelta
from pre_process import *
from post_process import *
from file_paths import *
from constants import *


def extract_params(route_params=False, demand=False, validation=False):
    if route_params:
        stops, ordered_trips, link_times_true, sched_deps_in, sched_arrs_in = get_route(path_stop_times, START_TIME_SEC,
                                                                                        END_TIME_SEC, TIME_NR_INTERVALS,
                                                                                        TIME_START_INTERVAL,
                                                                                        TIME_INTERVAL_LENGTH_MINS,
                                                                                        DATES,
                                                                                        TRIP_WITH_FULL_STOP_PATTERN)
        get_scheduled_bus_availability(path_stop_times, DATES, START_TIME_SEC, END_TIME_SEC)
        link_times_mean_true, link_times_sd_true, nr_time_dpoints_true = link_times_true
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_mean_true)
        save(path_ordered_trips, ordered_trips)
        save(path_departure_times_xtr, sched_deps_in)
        save('in/xtr/rt_20-2019-09/scheduled_arrivals_inbound.pkl', sched_arrs_in)
        sched_dep1_out, sched_dep2_out, sched_arr_out, trip_times1_params, trip_times2_params, \
        deadhead_time_params = get_outbound_travel_time(
            path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
            TRIP_TIME_INTERVAL_LENGTH_MINS)
        save('in/xtr/rt_20-2019-09/scheduled_departures_outbound1.pkl', sched_dep1_out)
        save('in/xtr/rt_20-2019-09/scheduled_departures_outbound2.pkl', sched_dep2_out)
        save('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl', sched_arr_out)
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
    return stops, link_times_mean, ordered_trips, scheduled_departures, odt, sched_arrivals, trip_times


def get_params_outbound():
    trip_times1_params = load('in/xtr/rt_20-2019-09/trip_time1_params.pkl')
    trip_times2_params = load('in/xtr/rt_20-2019-09/trip_time2_params.pkl')
    sched_deps1 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound1.pkl')
    sched_deps2 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound2.pkl')
    deadhead_times_params = load('in/xtr/rt_20-2019-09/deadhead_times_params.pkl')
    sched_arrs = load('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl')
    return trip_times1_params, trip_times2_params, sched_deps1, sched_deps2, deadhead_times_params, sched_arrs


# extract_params(route_params=True, validation=True)

STOPS, LINK_TIMES_MEAN, ORDERED_TRIPS, SCHED_DEP_IN, ODT, SCHED_ARRS_IN, TRIP_TIMES_INPUT = get_params_inbound()
TRIP_TIMES1_PARAMS, TRIP_TIMES2_PARAMS, SCHED_DEP_OUT1, SCHED_DEP_OUT2, DEADHEAD_TIME_PARAMS, SCHED_ARRS_OUT = get_params_outbound()
print([str(timedelta(seconds=x)) for x in SCHED_ARRS_OUT])
print([str(timedelta(seconds=x)) for x in SCHED_DEP_IN])

print([str(timedelta(seconds=x)) for x in SCHED_ARRS_IN])
print([str(timedelta(seconds=x)) for x in sorted(SCHED_DEP_OUT1 + SCHED_DEP_OUT2)])
sched_dep_out_id = np.array([1] * len(SCHED_DEP_OUT1) + [2] * len(SCHED_DEP_OUT2))
sched_dep_out_t = np.array(SCHED_DEP_OUT1 + SCHED_DEP_OUT2)
idx_sorted_t = sched_dep_out_t.argsort()
sorted_id = sched_dep_out_id[idx_sorted_t]
sorted_t = sched_dep_out_t[idx_sorted_t]
print(sorted_id.tolist())

print(DEADHEAD_TIME_PARAMS)




warm_up_odt = ODT[4]
for i in range(4):
    ODT = np.insert(ODT, 0, warm_up_odt, axis=0)
# for i in range(ODT.shape[0]):
#     plt.imshow(ODT[i])
#     plt.colorbar()
#     plt.savefig('in/vis/odt' + str(i) + '.png')
#     plt.close()
# SCHEDULED_DEPARTURES = UNIFORM_SCHEDULED_DEPARTURES.copy()
PAX_INIT_TIME = [0] + [LINK_TIMES_MEAN[s0 + '-' + s1][0] for s0, s1 in zip(STOPS, STOPS[1:])]
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_IN[0] - (SCHED_DEP_IN[1] - SCHED_DEP_IN[0])
# print([str(timedelta(seconds=i)) for i in SCHEDULED_DEPARTURES])
ordered_trips_arr = np.array([ORDERED_TRIPS])
scheduled_deps_arr = np.array([SCHED_DEP_IN])
FOCUS_TRIPS = ordered_trips_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
LAST_FOCUS_TRIP = FOCUS_TRIPS[-1]
# FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
UNIFORM_INTERVAL = 1
SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}
