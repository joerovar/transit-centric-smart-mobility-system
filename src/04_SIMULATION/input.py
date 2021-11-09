from pre_process import *
from post_process import *
from file_paths import *
from constants import *


def extract_params():
    stops, link_times_mean, link_times_sd, nr_time_dpoints, ordered_trips = get_route(path_stop_times,
                                                                                      SEC_FOR_TT_EXTRACT,
                                                                                      END_TIME_SEC,
                                                                                      TIME_NR_INTERVALS,
                                                                                      TIME_START_INTERVAL,
                                                                                      TIME_INTERVAL_LENGTH_MINS,
                                                                                      DATES,
                                                                                      TRIP_WITH_FULL_STOP_PATTERN,
                                                                                      path_ordered_dispatching,
                                                                                      path_sorted_daily_trips,
                                                                                      path_stop_pattern,
                                                                                      START_TIME_SEC,
                                                                                      FOCUS_START_TIME_SEC,
                                                                                      FOCUS_END_TIME_SEC)

    # arrival rates will be in pax/min
    arrival_rates, alight_fractions, alight_rates, dep_vol = get_demand(path_od, stops, PREV_DEM_NR_INTERVALS,
                                                                        PREV_DEM_START_INTERVAL, DEM_NR_INTERVALS,
                                                                        DEM_INTERVAL_LENGTH_MINS)

    # SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE
    # SCHEDULED_DEPARTURES = read_scheduled_departures(path_dispatching_times)
    scheduled_departures = get_dispatching_from_gtfs(path_ordered_dispatching, ordered_trips)

    save(path_alight_rates, alight_rates)
    save(path_dep_volume, dep_vol)
    save(path_route_stops, stops)
    save(path_link_times_mean, link_times_mean)
    save(path_link_times_sd, link_times_sd)
    save(path_link_dpoints, nr_time_dpoints)
    save(path_ordered_trips, ordered_trips)
    save(path_arr_rates, arrival_rates)
    save(path_alight_fractions, alight_fractions)
    save(path_departure_times_xtr, scheduled_departures)
    return


def get_params():
    stops = load(path_route_stops)
    link_times_mean = load(path_link_times_mean)
    link_times_sd = load(path_link_times_sd)
    nr_time_dpoints = load(path_link_dpoints)
    ordered_trips = load(path_ordered_trips)
    arrival_rates = load(path_arr_rates)
    alight_fractions = load(path_alight_fractions)
    scheduled_departures = load(path_departure_times_xtr)
    init_headway = scheduled_departures[1] - scheduled_departures[0]
    # initial headway helps calculate loads for the first trip
    return stops, link_times_mean, link_times_sd, nr_time_dpoints, ordered_trips, arrival_rates, alight_fractions, scheduled_departures, init_headway


def visualize_for_validation():
    # LOAD PROFILE
    arrival_rates = load(path_arr_rates)
    alight_rates = load(path_alight_rates)
    dep_volume = load(path_dep_volume)
    stops = load(path_route_stops)
    scheduled_departures = load(path_departure_times_xtr)
    avg_headway = scheduled_departures[1] - scheduled_departures[0]
    arrivals_per_trip = get_pax_per_trip(arrival_rates, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                         DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    alights_per_trip = get_pax_per_trip(alight_rates, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                        DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    dep_vol_per_trip = get_pax_per_trip(dep_volume, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                        DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    plot_load_profile(arrivals_per_trip, alights_per_trip, dep_vol_per_trip, stops, pathname=path_input_load_profile,
                      x_y_lbls=['stop id', 'pax per trip', 'pax load'])
    return


# extract_params()
# visualize_for_validation()

STOPS, LINK_TIMES_MEAN, LINK_TIMES_SD, NR_TIME_DPOINTS, ORDERED_TRIPS, ARRIVAL_RATES, ALIGHT_FRACTIONS, SCHEDULED_DEPARTURES, INIT_HEADWAY = get_params()
planned_headway_lst = [(j - i) for i, j in zip(SCHEDULED_DEPARTURES[:-1], SCHEDULED_DEPARTURES[1:])]
planned_headway_lbls = [str(i) + '-' + str(j) for i, j in zip(ORDERED_TRIPS[:-1], ORDERED_TRIPS[1:])]
PLANNED_HEADWAY = {i: j for i, j in zip(planned_headway_lbls, planned_headway_lst)}
