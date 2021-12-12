from datetime import datetime, timedelta, date
import numpy as np
# SIMULATION

START_TIME = datetime.strptime('07:00:00', "%H:%M:%S")
END_TIME = datetime.strptime('10:00:00', "%H:%M:%S")
T_TRAVEL_TIME_EXTRACT = datetime.strptime('06:00:00', "%H:%M:%S")
SEC_FOR_TT_EXTRACT = (T_TRAVEL_TIME_EXTRACT - datetime(1900, 1, 1)).total_seconds()
START_TIME_SEC = (START_TIME - datetime(1900, 1, 1)).total_seconds()
END_TIME_SEC = (END_TIME - datetime(1900, 1, 1)).total_seconds()
TOTAL_MIN = END_TIME - START_TIME
TOTAL_MIN = TOTAL_MIN.total_seconds() / 60
FOCUS_START_TIME = datetime.strptime('07:30:00', "%H:%M:%S")
FOCUS_END_TIME = datetime.strptime('08:30:00', "%H:%M:%S")
FOCUS_START_TIME_SEC = (FOCUS_START_TIME - datetime(1900, 1, 1)).total_seconds()
FOCUS_END_TIME_SEC = (FOCUS_END_TIME - datetime(1900, 1, 1)).total_seconds()
# ROUTE NETWORK: NUMBER OF ROUTES, NUMBER OF STOPS, ROUTE STOPS
# ROUTE 20 EAST
TIME_INTERVAL_LENGTH_MINS = 30
TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TIME_INTERVAL_LENGTH_MINS))
TIME_NR_INTERVALS = int(TOTAL_MIN / TIME_INTERVAL_LENGTH_MINS)
DATES = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']
TRIP_WITH_FULL_STOP_PATTERN = 911414030
# STARTING_TRIP = 911214020
# ENDING_TRIP = 910506020

# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
TTD = 'LOGNORMAL'
CV = 0.15
LOGN_S = np.sqrt(np.log(np.power(CV, 2) + 1))

ACC_DEC_TIME = 5
BOARDING_TIME = 3.0
ALIGHTING_TIME = 1.2

DDD = 'UNIFORM'
DEP_DELAY_FROM = -60
DEP_DELAY_TO = 90

# DEMAND: O-D POISSON RATES
PREV_DEM_INTERVAL_LENGTH_MINS = 5
PREV_DEM_START_INTERVAL = int(START_TIME_SEC / (60 * PREV_DEM_INTERVAL_LENGTH_MINS))
PREV_DEM_NR_INTERVALS = int(TOTAL_MIN / PREV_DEM_INTERVAL_LENGTH_MINS)
DEM_INTERVAL_LENGTH_MINS = 30
DEM_START_INTERVAL = int(START_TIME_SEC / (60 * DEM_INTERVAL_LENGTH_MINS))
DEM_END_INTERVAL = int(END_TIME_SEC / (60 * DEM_INTERVAL_LENGTH_MINS))
DEM_NR_INTERVALS = int(TOTAL_MIN / DEM_INTERVAL_LENGTH_MINS)
CAPACITY = 40
# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH
[IDX_ARR_T, IDX_DEP_T, IDX_LOAD, IDX_PICK, IDX_DROP, IDX_DENIED, IDX_HOLD_TIME, IDX_SKIPPED] = [i for i in range(1, 9)]
TPOINT0 = '386'
TPOINT1 = '8613'

# REINFORCEMENT LEARNING
BASE_HOLDING_TIME = 10
CONTROLLED_STOPS = ['390', '395', '14959', '14500', '413', '16110', '425', '431', '16049', '443']
LIMIT_HOLDING = 40
N_ACTIONS_RL = int(LIMIT_HOLDING/BASE_HOLDING_TIME) + 2
[IDX_RT_PROGRESS, IDX_LOAD_RL, IDX_FW_H, IDX_BW_H, IDX_PAX_AT_STOP] = [i for i in range(5)]
SKIP_ACTION = 0
