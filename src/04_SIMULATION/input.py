from pre_process import *
from post_process import *
from file_paths import *
from constants import *
from datetime import timedelta


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
    arrival_rates, alight_fractions, alight_rates, dep_vol, odt = get_demand(path_od, stops, PREV_DEM_NR_INTERVALS,
                                                                             PREV_DEM_START_INTERVAL, DEM_NR_INTERVALS,
                                                                             DEM_INTERVAL_LENGTH_MINS)

    # SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE
    # SCHEDULED_DEPARTURES = read_scheduled_departures(path_dispatching_times)
    scheduled_departures = get_dispatching_from_gtfs(path_ordered_dispatching, ordered_trips)

    save(path_odt_xtr, odt)
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
    odt = load(path_odt_xtr)
    # initial headway helps calculate loads for the first trip
    return stops, link_times_mean, link_times_sd, nr_time_dpoints, ordered_trips, arrival_rates, alight_fractions, scheduled_departures, init_headway, odt


def visualize_for_validation():
    # LOAD PROFILE
    arrival_rates = load(path_arr_rates)
    alight_rates = load(path_alight_rates)
    dep_volume = load(path_dep_volume)
    stops = load(path_route_stops)
    scheduled_departures = load(path_departure_times_xtr)
    avg_headway = scheduled_departures[1] - scheduled_departures[0]
    ordered_trips = load(path_ordered_trips)
    link_times_mean = load(path_link_times_mean)
    link_times_sd = load(path_link_times_sd)

    arrivals_per_trip = get_pax_per_trip(arrival_rates, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                         DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    alights_per_trip = get_pax_per_trip(alight_rates, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                        DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    dep_vol_per_trip = get_pax_per_trip(dep_volume, FOCUS_START_TIME_SEC / 3600, FOCUS_END_TIME_SEC / 3600,
                                        DEM_START_INTERVAL, DEM_INTERVAL_LENGTH_MINS / 60, avg_headway / 3600)
    plot_load_profile(arrivals_per_trip, alights_per_trip, dep_vol_per_trip, stops, pathname=path_input_load_profile,
                      x_y_lbls=['stop id', 'pax per trip', 'pax load'])

    ordered_trips_array = np.array(ordered_trips)
    scheduled_departures_array = np.array(scheduled_departures)
    subset_ordered_trips = ordered_trips_array[(scheduled_departures_array >= START_TIME_SEC) & (
            scheduled_departures_array <= FOCUS_END_TIME_SEC)].tolist()
    historical_headway = get_historical_headway(path_stop_times, DATES, stops, subset_ordered_trips,
                                                FOCUS_START_TIME_SEC, FOCUS_END_TIME_SEC)
    plot_headway(path_historical_headway, historical_headway, stops)

    single_link_time_mean = {}
    single_link_time_sd = {}
    interval_focus = 1
    for link in link_times_mean:
        single_link_time_mean[link] = link_times_mean[link][interval_focus]
        single_link_time_sd[link] = link_times_sd[link][interval_focus]
    ltimes_lbl = ['mean', 'stdev']
    ltimes_x_y_lbls = ['stops', 'seconds']
    plot_link_times(single_link_time_mean, single_link_time_sd, stops, path_input_link_times_fig, ltimes_lbl,
                    x_y_lbls=ltimes_x_y_lbls)
    return


# extract_params()
visualize_for_validation()

# STOPS, LINK_TIMES_MEAN, LINK_TIMES_SD, NR_TIME_DPOINTS, ORDERED_TRIPS, ARRIVAL_RATES, ALIGHT_FRACTIONS, SCHEDULED_DEPARTURES, INIT_HEADWAY, ODT = get_params()
# planned_headway_lst = [(j - i) for i, j in zip(SCHEDULED_DEPARTURES[:-1], SCHEDULED_DEPARTURES[1:])]
# planned_headway_lbls = [str(i) + '-' + str(j) for i, j in zip(ORDERED_TRIPS[:-1], ORDERED_TRIPS[1:])]
# PLANNED_HEADWAY = {i: j for i, j in zip(planned_headway_lbls, planned_headway_lst)}
#
# # FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
# CONSTANT_HEADWAY = 270
# UNIFORM_SCHEDULED_DEPARTURES = [SCHEDULED_DEPARTURES[0] + CONSTANT_HEADWAY * i for i in
#                                 range(len(SCHEDULED_DEPARTURES))]
# UNIFORM_INTERVAL = 1
# ARRIVAL_RATE = {key: value[UNIFORM_INTERVAL] for (key, value) in ARRIVAL_RATES.items()}
# SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}
# ALIGHT_FRACTION = {key: value[UNIFORM_INTERVAL] for (key, value) in ALIGHT_FRACTIONS.items()}

# print([str(timedelta(seconds=i)) for i in SCHEDULED_DEPARTURES])
# print(planned_headway_lst)
# print(ORDERED_TRIPS)
