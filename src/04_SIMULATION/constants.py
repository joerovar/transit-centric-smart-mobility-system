from datetime import datetime
import math

# SIMULATION

START_TIME = datetime.strptime('05:00:00', "%H:%M:%S")
END_TIME = datetime.strptime('10:00:00', "%H:%M:%S")
START_TIME_SEC = (START_TIME - datetime(1900, 1, 1)).total_seconds()
END_TIME_SEC = (END_TIME - datetime(1900, 1, 1)).total_seconds()
TOTAL_MIN = (END_TIME - START_TIME).total_seconds() / 60
FOCUS_START_TIME = datetime.strptime('07:28:00', "%H:%M:%S")
FOCUS_END_TIME = datetime.strptime('08:15:00', "%H:%M:%S")
FOCUS_START_TIME_SEC = (FOCUS_START_TIME - datetime(1900, 1, 1)).total_seconds()
FOCUS_END_TIME_SEC = (FOCUS_END_TIME - datetime(1900, 1, 1)).total_seconds()
# ROUTE NETWORK: NUMBER OF ROUTES, NUMBER OF STOPS, ROUTE STOPS
DELAY_INTERVAL_LENGTH_MINS = 60
DELAY_START_INTERVAL = int(START_TIME_SEC / (60 * DELAY_INTERVAL_LENGTH_MINS))
# ROUTE 20 EAST
TIME_INTERVAL_LENGTH_MINS = 30
TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TIME_INTERVAL_LENGTH_MINS))
TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TIME_INTERVAL_LENGTH_MINS))
# DATES = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']
DATES = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06',
         '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12', '2019-09-13',
         '2019-09-16', '2019-09-17', '2019-09-18', '2019-09-19', '2019-09-20',
         '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27']
TRIP_WITH_FULL_STOP_PATTERN = 911414030
# INBOUND TO OUTBOUND LAYOVER TIME
MIN_LAYOVER_T = 0
ERR_LAYOVER_TIME = 0
# INBOUND TIME DEPENDENT TRIP TIME DISTRIBUTION
TRIP_TIME_INTERVAL_LENGTH_MINS = 60
TRIP_TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TRIP_TIME_INTERVAL_LENGTH_MINS))
TRIP_TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TRIP_TIME_INTERVAL_LENGTH_MINS))
# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
# NOT TUNED
ACC_DEC_TIME = 3.5
BOARDING_TIME = 2.1
ALIGHTING_TIME = 1.3
DWELL_TIME_ERROR = 2.2
EXTREME_TT_BOUND = 0.95
CAPACITY = 50
DDD = 'UNIFORM'
DEP_DELAY_FROM = -60
DEP_DELAY_TO = 110

# DEMAND: NEW O-D RATES FROM DINGYI
ODT_INTERVAL_LEN_MIN = 30
ODT_START_INTERVAL = int(START_TIME_SEC / (60 * ODT_INTERVAL_LEN_MIN))
ODT_END_INTERVAL = int(END_TIME_SEC / (60 * ODT_INTERVAL_LEN_MIN))

# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH
[IDX_ARR_T, IDX_DEP_T, IDX_LOAD, IDX_PICK, IDX_DROP, IDX_DENIED, IDX_HOLD_TIME, IDX_SKIPPED, IDX_SCHED] = [i for i in
                                                                                                           range(1, 10)]
NO_OVERTAKE_BUFFER = 5

# REINFORCEMENT LEARNING

CONTROLLED_STOPS = ['386', '409', '423', '16049', '3954']
CONTROLLED_STOPS_ALTERNATIVE = ['386', '409', '428', '3954']
N_STATE_PARAMS_RL = 6
[IDX_RT_PROGRESS, IDX_LOAD_RL, IDX_FW_H, IDX_BW_H, IDX_PAX_AT_STOP, IDX_PREV_FW_H] = [i for i in
                                                                                      range(N_STATE_PARAMS_RL)]
SKIP_ACTION = 0
# WEIGHT_WAIT_TIME = 1.4

INBOUND_SHORT_START_STOP = '15136'
INBOUND_LONG_START_STOP = '8613'
[IDX_ARR_T_IN, IDX_SCHED_IN] = [i for i in range(1, 3)]

OUT_TRIP_RECORD_COLS = ['trip_id', 'stop_id', 'arr_sec', 'dep_sec', 'pax_load', 'ons', 'offs', 'denied',
                        'hold_time', 'skipped', 'schd_sec', 'stop_sequence'] # and replication goes at the end
IN_TRIP_RECORD_COLS = ['trip_id', 'stop_id', 'arr_sec', 'schd_sec', 'stop_sequence']
PAX_RECORD_COLS = ['orig_idx', 'dest_idx', 'arr_time', 'board_time', 'alight_time', 'trip_id', 'denied']
