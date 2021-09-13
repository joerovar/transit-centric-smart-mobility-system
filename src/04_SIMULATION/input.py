from extract_tools import *
from file_paths import *
from const import *
import csv

STOPS, LINK_TIMES_MEAN, LINK_TIMES_STD, NR_TIME_DPOINTS, ORDERED_TRIPS = get_route(path_stop_times,
                                                                                   SEC_FOR_TT_EXTRACT, END_TIME_SEC,
                                                                                   TIME_NR_INTERVALS,
                                                                                   TIME_START_INTERVAL,
                                                                                   TIME_INTERVAL_LENGTH_MINS,
                                                                                   DATES,
                                                                                   TRIP_WITH_FULL_STOP_PATTERN)

start_idx = ORDERED_TRIPS.index(STARTING_TRIP)
end_idx = ORDERED_TRIPS.index(ENDING_TRIP)
ORDERED_TRIPS = ORDERED_TRIPS[start_idx:end_idx+1]


# arrival rates will be in pax/min
ARRIVAL_RATES, ALIGHT_FRACTIONS = get_demand(path_od, STOPS, DEM_NR_INTERVALS,
                                             DEM_INTERVAL_LENGTH_MINS, DEM_START_INTERVAL)

# SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE

# SCHEDULED_DEPARTURES = read_scheduled_departures(path_dispatching_times)
SCHEDULED_DEPARTURES = get_dispatching_from_gtfs('in/ordered_dispatching.csv', ORDERED_TRIPS)
INIT_HEADWAY = SCHEDULED_DEPARTURES[1] - SCHEDULED_DEPARTURES[0]
# initial headway helps calculate loads for the first trip

# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH

CAPACITY = 40

# FOR HEADWAY CALCULATION: PERIOD OF HIGH FREQUENCY WITH FULL VARIATION OF ROUTE


# hw = get_historical_headway(path_stop_times, DATES, STOPS, ORDERED_TRIPS)
# plot_stop_headway(hw, 'in/historical_hw.png')
# plot_cv('in/cv_link_times.png', LINK_TIMES_MEAN, LINK_TIMES_STD)
# write_travel_times(path_link_times, LINK_TIMES_MEAN, LINK_TIMES_STD, NR_TIME_DPOINTS)
