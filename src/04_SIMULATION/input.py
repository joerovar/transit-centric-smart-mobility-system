import pandas as pd

from pre_process import *
from post_process import *
from file_paths import *
from constants import *
from datetime import timedelta


def extract_params(route_params=False, demand=False, validation=False):
    if route_params:
        stops, ordered_trips, link_times_true = get_route(path_stop_times, SEC_FOR_TT_EXTRACT, END_TIME_SEC,
                                                          TIME_NR_INTERVALS, TIME_START_INTERVAL,
                                                          TIME_INTERVAL_LENGTH_MINS, DATES,
                                                          TRIP_WITH_FULL_STOP_PATTERN,
                                                          path_ordered_dispatching,
                                                          path_sorted_daily_trips, path_stop_pattern,
                                                          START_TIME_SEC, FOCUS_START_TIME_SEC,
                                                          FOCUS_END_TIME_SEC)

        scheduled_departures_in = get_dispatching_from_gtfs(path_ordered_dispatching, ordered_trips)
        get_scheduled_bus_availability(path_stop_times, DATES, START_TIME_SEC, END_TIME_SEC)
        link_times_mean_true, link_times_sd_true, nr_time_dpoints_true = link_times_true
        save(path_route_stops, stops)
        save(path_link_times_mean, link_times_mean_true)
        save(path_ordered_trips, ordered_trips)
        save(path_departure_times_xtr, scheduled_departures_in)
        sched_dep1_out, sched_dep2_out, sched_arr_out, delay1_params, \
        triptime1_params, delay2_params, triptime2_params, arrival_hw_out = get_outbound_travel_time(
            path_stop_times,
            START_TIME_SEC,
            END_TIME_SEC, DATES)
        save('in/xtr/rt_20-2019-09/scheduled_departures_outbound1.pkl', sched_dep1_out)
        save('in/xtr/rt_20-2019-09/scheduled_departures_outbound2.pkl', sched_dep2_out)
        save('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl', sched_arr_out)
        save('in/xtr/rt_20-2019-09/dep_delay1_params.pkl', delay1_params)
        save('in/xtr/rt_20-2019-09/dep_delay2_params.pkl', delay2_params)
        save('in/xtr/rt_20-2019-09/trip_time1_params.pkl', triptime1_params)
        save('in/xtr/rt_20-2019-09/trip_time2_params.pkl', triptime2_params)
        save('in/xtr/rt_20-2019-09/arrival_headway_outbound.pkl', arrival_hw_out)
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
        # schedule = [schedule[0] + HEADWAY_UNIFORM * i for i in range(len(schedule))]
        schedule_arr = np.array(schedule)
        trip_ids_arr = np.array(trip_ids)
        focus_trips = trip_ids_arr[
            (schedule_arr <= FOCUS_END_TIME_SEC) & (schedule_arr >= FOCUS_START_TIME_SEC)].tolist()
        trip_times, departure_headway_in = get_trip_times(path_stop_times, focus_trips, DATES, START_TIME_SEC,
                                                          END_TIME_SEC)
        save('in/xtr/rt_20-2019-09/trip_times.pkl', trip_times)
        dwell_times_mean, dwell_times_std, dwell_times_tot = get_dwell_times(path_stop_times, focus_trips, stops, DATES)
        save('in/xtr/rt_20-2019-09/dwell_times_mean.pkl', dwell_times_mean)
        save('in/xtr/rt_20-2019-09/dwell_times_std.pkl', dwell_times_std)
        save('in/xtr/rt_20-2019-09/dwell_times_tot.pkl', dwell_times_tot)
        save('in/xtr/rt_20-2019-09/departure_headway_inbound.pkl', departure_headway_in)
        write_inbound_trajectories(path_stop_times, ordered_trips)
        load_profile = get_load_profile(path_stop_times, focus_trips, stops)
        save('in/xtr/rt_20-2019-09/load_profile.pkl', load_profile)
    return


def get_params():
    stops = load(path_route_stops)
    link_times_mean = load(path_link_times_mean)
    ordered_trips = load(path_ordered_trips)
    scheduled_departures = load(path_departure_times_xtr)
    init_headway = scheduled_departures[1] - scheduled_departures[0]
    odt = load(path_odt_xtr)
    # initial headway helps calculate loads for the first trip
    return stops, link_times_mean, ordered_trips, scheduled_departures, init_headway, odt


extract_params(demand=True)

STOPS, LINK_TIMES_MEAN, ORDERED_TRIPS, SCHEDULED_DEPARTURES, INIT_HEADWAY, ODT = get_params()
for i in range(ODT.shape[0]):
    plt.imshow(ODT[i])
    plt.colorbar()
    plt.savefig('in/vis/odt' + str(i) + '.png')
    plt.close()
# scheduled_deps_arr = np.array(SCHEDULED_DEPARTURES)
# focus_dep = scheduled_deps_arr[(scheduled_deps_arr <= FOCUS_END_TIME_SEC)].tolist()
# focus_headway = [i-j for i, j in zip(focus_dep[1:], focus_dep[:-1])]
# focus_avg_headway = np.mean(focus_headway)
UNIFORM_SCHEDULED_DEPARTURES = [SCHEDULED_DEPARTURES[0] + HEADWAY_UNIFORM * i for i in
                                range(len(SCHEDULED_DEPARTURES))]
# SCHEDULED_DEPARTURES = UNIFORM_SCHEDULED_DEPARTURES.copy()
PAX_INIT_TIME = [0] + [LINK_TIMES_MEAN[s0 + '-' + s1][0] for s0, s1 in zip(STOPS, STOPS[1:])]
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHEDULED_DEPARTURES[0] - INIT_HEADWAY
# print([str(timedelta(seconds=i)) for i in SCHEDULED_DEPARTURES])
ordered_trips_arr = np.array([ORDERED_TRIPS])
scheduled_deps_arr = np.array([SCHEDULED_DEPARTURES])
FOCUS_TRIPS = ordered_trips_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
LAST_FOCUS_TRIP = FOCUS_TRIPS[-1]
# FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
UNIFORM_INTERVAL = 1
SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}

