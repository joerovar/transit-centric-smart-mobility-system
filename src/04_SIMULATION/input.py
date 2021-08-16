from input_fns import *
from datetime import datetime, timedelta
import csv
# SIMULATION

INIT_SIM_TIME = datetime.strptime('07:00:00', "%H:%M:%S")
FIN_SIM_TIME = datetime.strptime('10:00:00', "%H:%M:%S")
# ROUTE NETWORK: NUMBER OF ROUTES, NUMBER OF STOPS, ROUTE STOPS
# notes:
# trips.txt records the id of all trips for a route
# stop_times.txt records all arr and dep times at all stops for a specific trip id
ROUTE_ID = '20'
ROUTE_DIRECTION = 'East'
STOPS, LINK_TIMES_MEAN, LINK_TIMES_STDEV = get_stops(ROUTE_ID, ROUTE_DIRECTION,
                                                     'in/gtfs/trips.txt', 'in/route20_stop_time.dat',
                                                     INIT_SIM_TIME, FIN_SIM_TIME)
# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
TTD = 'LOGNORMAL'
CV = 0.30
LOGN_S = np.sqrt(np.log(np.power(CV, 2)+1))

STOPPING_DELAY = 3
BOARDING_DELAY = 2
ALIGHTING_DELAY = 1

DDD = 'UNIFORM'
DEP_DELAY_FROM = -60
DEP_DELAY_TO = 120

# DEMAND: O-D POISSON RATES
# ARRIVAL_RATES, ALIGHT_FRACTIONS, PEAK_VOL = extract_demand('in/od.dat', ROUTE_STOPS)
NR_INTERVALS = int(180/5)
INTERVAL_LENGTH_MINS = 5
START_INTERVAL = 84
# arrival rates will be in pax/min
ARRIVAL_RATES, ALIGHT_FRACTIONS = get_demand('in/odt_for_opt.dat', STOPS, NR_INTERVALS, INTERVAL_LENGTH_MINS, START_INTERVAL)

# SCHEDULE: DISPATCHING TIMES, STOP TIMES FROM BEGINNING OF ROUTE
SCHEDULED_DEPARTURES = read_scheduled_departures('in/dispatching_time.dat')
INIT_HEADWAY = SCHEDULED_DEPARTURES[1] - SCHEDULED_DEPARTURES[0]
# initial headway helps calculate loads for the first trip

# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH
START_SIMUL_TIME = 7 * 60.0 * 60.0
STOP_SIMUL_TIME = 10 * 60.0 * 60.0
FINAL_TRIP_DEPARTURE = SCHEDULED_DEPARTURES[-1]
CAPACITY = 25
DELAY_TOLERANCE = 2.0 * 60

# CHECK

# SEE DISTRIBUTION
NMEAN = 2.5
NMEAN_REQ = 1.9
# time units in seconds

