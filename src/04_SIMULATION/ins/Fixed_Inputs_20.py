from datetime import datetime
import math

# route
FULL_PATTERN_HEADSIGN = 'Illinois Center'
RT_NR = 20
OB_DIRECTION = 'East'
IB_DIRECTION = 'West'
DIR_ROUTE = 'ins/rt_20_2019-09/'
DIR_ROUTE_OUTS = 'outs/rt_20_2019-09/'

# time period
START_TIME = datetime.strptime('05:00:00', "%H:%M:%S")
END_TIME = datetime.strptime('10:00:00', "%H:%M:%S")

# dates for avl/apc data collection
y_mo = '2019-09-'
days = list(range(3, 7)) + list(range(9, 14)) + list(range(16, 21)) + list(range(23, 28))
days_str = ['0' + str(d) if len(str(d)) == 1 else str(d) for d in days]
DATES = [y_mo + d for d in days_str]

START_TIME_SEC = (START_TIME - datetime(1900, 1, 1)).total_seconds()
END_TIME_SEC = (END_TIME - datetime(1900, 1, 1)).total_seconds()
TOTAL_MIN = (END_TIME - START_TIME).total_seconds() / 60
DELAY_BIN_MINS = 60
TIME_BIN_MINS = 30
ODT_BIN_MINS = 30
TRIP_TIME_BIN_MINS = 60

# INBOUND TO OUTBOUND LAYOVER TIME
MIN_LAYOVER_T = 50
ERR_LAYOVER_TIME = 20
# INBOUND TIME DEPENDENT TRIP TIME DISTRIBUTION


# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
ACC_DEC_TIME = 4.5
BOARDING_TIME = 2.3
ALIGHTING_TIME = 1.2
DWELL_TIME_ERROR = 3.0
EXTREME_TT_BOUND = 1.0
BOOST_SCHED_RUN_T = 1.3 # FOR THOSE PROBLEMATIC LINKS (FIRST AND LAST) INCREASE SCHED RUN TIME BY 30 PCT
DEP_DELAY_FROM = -60
DEP_DELAY_TO = 110

# OTHER SERVICE PARAMETERS: DWELL TIME, SIMULATION LENGTH
[IDX_ARR_T, IDX_DEP_T, IDX_LOAD, IDX_PICK, IDX_DROP, IDX_DENIED, IDX_HOLD_TIME, IDX_SKIPPED, IDX_SCHED] = [i for i in
                                                                                                           range(1, 10)]
HIGH_CAPACITY = 80
LOW_CAPACITY = 53

# FOR RECORDS
OUT_TRIP_RECORD_COLS = ['bus_id', 'trip_id', 'stop_id', 'arr_sec', 'dep_sec', 'pax_load', 'ons', 'offs', 'denied',
                        'hold_time', 'skipped', 'schd_sec', 'stop_sequence', 'dist_traveled', 'expressed'] # and replication goes at the end
IN_TRIP_RECORD_COLS = ['trip_id', 'stop_id', 'arr_sec', 'schd_sec', 'stop_sequence']
PAX_RECORD_COLS = ['orig_idx', 'dest_idx', 'arr_time', 'board_time', 'alight_time', 'trip_id', 'denied']

# TERMINAL DISPATCHING PARAMS
EARLY_DEP_LIMIT_SEC = 120 # seconds
IMPOSED_DELAY_LIMIT = 180
HOLD_INTERVALS = 60
FUTURE_HW_HORIZON = 2
PAST_HW_HORIZON = 2
END_TIME_SEC2 = END_TIME_SEC - 60*30
WEIGHT_HOLD_T = 0.0
EXPRESS_DIST = 9
BW_H_LIMIT_EXPRESS = 0.5 # this is the limit over the scheduled backward headway to decide to express
FW_H_LIMIT_EXPRESS = 1.7 # this is the limit over the scheduled forward headway to decide to express
BF_H_LIMIT_EXPRESS = 1.6
MRH_HOLD_LIMIT = 90

# GENERAL RL PARAMS
LEARN_RATE = 0.0009
DISCOUNT_FACTOR = 0.99

# DDQN PARAMS
EPS_MIN = 0.01
EPS_DEC = 0.00006
EPS = 0.99
MAX_MEM = 600
BATCH_SIZE = 16
EPOCHS_REPLACE = 300
FC_DIMS = 256
ALGO = 'DDQNAgent'


# PARAMS FOR DISPATCHING CONTROL WITH RL
NR_STATE_D_RL = FUTURE_HW_HORIZON*2 + PAST_HW_HORIZON*2 + 1 # actual, scheduled and sched deviation
NR_ACTIONS_D_RL = int((IMPOSED_DELAY_LIMIT+EARLY_DEP_LIMIT_SEC)/HOLD_INTERVALS) + 1
# PARAMS FOR AT-STOP CONTROL WITH RL


N_STATE_PARAMS_RL = 6
[IDX_RT_PROGRESS, IDX_LOAD_RL, IDX_FW_H, IDX_BW_H, IDX_PAX_AT_STOP, IDX_PREV_FW_H] = [i for i in
                                                                                      range(N_STATE_PARAMS_RL)]
SKIP_ACTION = 0
ESTIMATED_PAX = False
WEIGHT_RIDE_T = 0.0
TT_FACTOR = 1.0
HOLD_ADJ_FACTOR = 0.0

# MID-ROUTE HOLDING STOPS
MRH_STOPS = ['415', '431']

# HOLDING AND STOP-SKIPPING EXPERIMENTS
CONTROL_STOPS = ['386', '409', '423', '16049']
TERMINAL_STATE_STOP = '3954'
BASE_HOLDING_TIME = 25
MIN_HW_THRESHOLD = 0.4

# TRIP FOCUSED ANALYSIS PERIOD
FOCUS_START_TIME = datetime.strptime('07:28:00', "%H:%M:%S")
FOCUS_END_TIME = datetime.strptime('08:15:00', "%H:%M:%S")

# INTERVALS
DELAY_START_INTERVAL = int(START_TIME_SEC / (60 * DELAY_BIN_MINS))
TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TIME_BIN_MINS))
TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TIME_BIN_MINS))
TRIP_TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TRIP_TIME_BIN_MINS))
TRIP_TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TRIP_TIME_BIN_MINS))
ODT_START_INTERVAL = int(START_TIME_SEC / (60 * ODT_BIN_MINS))
ODT_END_INTERVAL = int(END_TIME_SEC / (60 * ODT_BIN_MINS))
