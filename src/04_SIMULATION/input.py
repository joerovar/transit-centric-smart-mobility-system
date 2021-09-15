from extract_tools import *
from data_tools import *
from file_paths import *
from const import *


def extract_params(visualize=True):
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
                                                                                      path_stop_pattern)

    # arrival rates will be in pax/min
    arrival_rates, alight_fractions = get_demand(path_od, stops, PREV_DEM_NR_INTERVALS,
                                                 PREV_DEM_INTERVAL_LENGTH_MINS, PREV_DEM_START_INTERVAL, new_nr_intervals=DEM_NR_INTERVALS)

    # SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE
    # SCHEDULED_DEPARTURES = read_scheduled_departures(path_dispatching_times)
    scheduled_departures = get_dispatching_from_gtfs(path_ordered_dispatching, ordered_trips)

    save(path_route_stops, stops)
    save(path_link_times_mean, link_times_mean)
    save(path_link_times_sd, link_times_sd)
    save(path_link_dpoints, nr_time_dpoints)
    save(path_ordered_trips, ordered_trips)
    save(path_arr_rates, arrival_rates)
    save(path_alight_fractions, alight_fractions)
    save(path_departure_times_xtr, scheduled_departures)
    if visualize:
        hw = get_historical_headway(path_stop_times, DATES, stops, ordered_trips)
        plot_stop_headway(path_historical_headway, hw, stops)
        plot_cv(path_input_cv_link_times, link_times_mean, link_times_sd)
        write_travel_times(path_input_link_times, link_times_mean, link_times_sd, nr_time_dpoints)
        boardings = get_input_boardings(arrival_rates, DEM_INTERVAL_LENGTH_MINS, FOCUS_START_TIME_SEC, FOCUS_END_TIME_SEC, DEM_START_INTERVAL)
        plot_pax_per_stop(path_input_boardings, boardings, stops, x_y_lbls=['stop id', 'predicted boardings'])
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

    start_idx = ordered_trips.index(STARTING_TRIP)
    end_idx = ordered_trips.index(ENDING_TRIP)
    ordered_trips = ordered_trips[start_idx:end_idx + 1]
    init_headway = scheduled_departures[1] - scheduled_departures[0]
    # initial headway helps calculate loads for the first trip

    return stops, link_times_mean, link_times_sd, nr_time_dpoints, ordered_trips, arrival_rates, alight_fractions, scheduled_departures, init_headway


# extract_params()

STOPS, LINK_TIMES_MEAN, LINK_TIME_SD, NR_TIME_DPOINTS, ORDERED_TRIPS, ARRIVAL_RATES, ALIGHT_FRACTIONS, SCHEDULED_DEPARTURES, INIT_HEADWAY = get_params()
print(ALIGHT_FRACTIONS)

