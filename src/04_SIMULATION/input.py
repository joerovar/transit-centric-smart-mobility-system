from input_fns import *
from datetime import datetime, timedelta, date
from file_paths import *
import csv

# SIMULATION

START_TIME = datetime.strptime('07:00:00', "%H:%M:%S")
END_TIME = datetime.strptime('10:00:00', "%H:%M:%S")
T_TRAVEL_TIME_EXTRACT = datetime.strptime('06:00:00', "%H:%M:%S")
SEC_FOR_TT_EXTRACT = (T_TRAVEL_TIME_EXTRACT - datetime(1900, 1, 1)).total_seconds()
START_TIME_SEC = (START_TIME - datetime(1900, 1, 1)).total_seconds()
END_TIME_SEC = (END_TIME - datetime(1900, 1, 1)).total_seconds()
TOTAL_MIN = END_TIME - START_TIME
TOTAL_MIN = TOTAL_MIN.total_seconds() / 60
# ROUTE NETWORK: NUMBER OF ROUTES, NUMBER OF STOPS, ROUTE STOPS
# ROUTE 20 EAST
TIME_INTERVAL_LENGTH_MINS = 30
TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TIME_INTERVAL_LENGTH_MINS))
TIME_NR_INTERVALS = int(TOTAL_MIN / TIME_INTERVAL_LENGTH_MINS)
DATES = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']
TRIP_WITH_FULL_STOP_PATTERN = 911414030
STOPS, LINK_TIMES_MEAN, LINK_TIMES_STD, NR_TIME_DPOINTS, ORDERED_TRIPS = get_route(path_stop_times,
                                                                                   SEC_FOR_TT_EXTRACT, END_TIME_SEC,
                                                                                   TIME_NR_INTERVALS,
                                                                                   TIME_START_INTERVAL,
                                                                                   TIME_INTERVAL_LENGTH_MINS,
                                                                                   DATES,
                                                                                   TRIP_WITH_FULL_STOP_PATTERN)
STARTING_TRIP = 911214020
ENDING_TRIP = 910506020
start_idx = ORDERED_TRIPS.index(STARTING_TRIP)
end_idx = ORDERED_TRIPS.index(ENDING_TRIP)
ORDERED_TRIPS = ORDERED_TRIPS[start_idx:end_idx+1]

# hw = get_historical_headway(path_stop_times, DATES, STOPS, ORDERED_TRIPS)
# plot_stop_headway(hw, 'in/historical_hw.png')
# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
TTD = 'LOGNORMAL'
CV = 0.30
LOGN_S = np.sqrt(np.log(np.power(CV, 2) + 1))

ACC_DEC_TIME = 5
BOARDING_TIME = 3.1
ALIGHTING_TIME = 2.1

DDD = 'UNIFORM'
DEP_DELAY_FROM = -60
DEP_DELAY_TO = 120

# DEMAND: O-D POISSON RATES
DEM_INTERVAL_LENGTH_MINS = 5
DEM_START_INTERVAL = int(START_TIME_SEC / (60 * DEM_INTERVAL_LENGTH_MINS))
DEM_NR_INTERVALS = int(TOTAL_MIN / DEM_INTERVAL_LENGTH_MINS)

# arrival rates will be in pax/min
ARRIVAL_RATES, ALIGHT_FRACTIONS = get_demand(path_od, STOPS, DEM_NR_INTERVALS,
                                             DEM_INTERVAL_LENGTH_MINS, DEM_START_INTERVAL)

# SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE
# SCHEDULED_DEPARTURES = read_scheduled_departures(path_dispatching_times)
# print(SCHEDULED_DEPARTURES)
SCHEDULED_DEPARTURES = get_dispatching_from_gtfs('in/ordered_dispatching.csv', ORDERED_TRIPS)
INIT_HEADWAY = SCHEDULED_DEPARTURES[1] - SCHEDULED_DEPARTURES[0]
# initial headway helps calculate loads for the first trip

# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH

CAPACITY = 40

# FOR HEADWAY CALCULATION: PERIOD OF HIGH FREQUENCY WITH FULL VARIATION OF ROUTE

plot_cv('in/cv_link_times.png', LINK_TIMES_MEAN, LINK_TIMES_STD)
# write_travel_times(path_link_times, LINK_TIMES_MEAN, LINK_TIMES_STD, NR_TIME_DPOINTS)
