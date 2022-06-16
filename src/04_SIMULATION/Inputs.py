import numpy as np
import pandas as pd
from Input_Processor import extract_outbound_params, extract_inbound_params, extract_demand
from File_Paths import *
from datetime import datetime
import math
from Output_Processor import load
from datetime import timedelta

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
MIN_LAYOVER_T = 40
ERR_LAYOVER_TIME = 20
# INBOUND TIME DEPENDENT TRIP TIME DISTRIBUTION
TRIP_TIME_INTERVAL_LENGTH_MINS = 60
TRIP_TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TRIP_TIME_INTERVAL_LENGTH_MINS))
TRIP_TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TRIP_TIME_INTERVAL_LENGTH_MINS))
# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
# NOT TUNED
ACC_DEC_TIME = 5.0
BOARDING_TIME = 2.5
ALIGHTING_TIME = 1.5
DWELL_TIME_ERROR = 3.0
EXTREME_TT_BOUND = 1.00
CAPACITY = 60
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

LEARN_RATE = 0.0001
EPS_MIN = 0.01
DISCOUNT_FACTOR = 0.985
EPS_DEC = 1.5e-5
EPS = 0.64
MAX_MEM = 8000
BATCH_SIZE = 32
EPISODES_REPLACE = 600
FC_DIMS = 256
ALGO = 'DDQNAgent'
WEIGHT_RIDE_T = 0.0
TT_FACTOR = 1.0
HOLD_ADJ_FACTOR = 0.0
ESTIMATED_PAX = False
NETS_PATH = 'out/trained_nets/'

INBOUND_SHORT_START_STOP = '15136'
INBOUND_LONG_START_STOP = '8613'
[IDX_ARR_T_IN, IDX_SCHED_IN] = [i for i in range(1, 3)]

OUT_TRIP_RECORD_COLS = ['bus_id', 'trip_id', 'stop_id', 'arr_sec', 'dep_sec', 'pax_load', 'ons', 'offs', 'denied',
                        'hold_time', 'skipped', 'schd_sec', 'stop_sequence', 'dist_traveled'] # and replication goes at the end
IN_TRIP_RECORD_COLS = ['trip_id', 'stop_id', 'arr_sec', 'schd_sec', 'stop_sequence']
PAX_RECORD_COLS = ['orig_idx', 'dest_idx', 'arr_time', 'board_time', 'alight_time', 'trip_id', 'denied']

# TERMINAL DISPATCHING PARAMS
EARLY_DEP_LIMIT_SEC = 60 # seconds
MAX_DELAY_FOR_CONTROL = 120 # seconds
IMPOSED_DELAY_LIMIT = 90
HOLD_INTERVALS = 30
PROB_CANCELLED_BLOCK = 0.0
CANCEL_NOTICE_TIME = 10*60 # seconds
FUTURE_HW_HORIZON = 2
PAST_HW_HORIZON = 2
# extract_demand(ODT_INTERVAL_LEN_MIN, DATES)
# extract_outbound_params(START_TIME_SEC, END_TIME_SEC, TIME_NR_INTERVALS,
#                         TIME_START_INTERVAL, TIME_INTERVAL_LENGTH_MINS, DATES,
#                         TRIP_WITH_FULL_STOP_PATTERN, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL)
# extract_inbound_params(
#     START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
#     TRIP_TIME_INTERVAL_LENGTH_MINS, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL)

# OUTBOUND
LINK_TIMES_INFO = load(path_link_times_mean)
TRIPS_OUT_INFO = load('in/xtr/trips_outbound_info.pkl')
ODT_RATES_SCALED = np.load('in/xtr/rt_20_odt_rates_30_scaled.npy')
ODT_STOP_IDS = list(np.load('in/xtr/rt_20_odt_stops.npy'))
ODT_STOP_IDS = [str(int(s)) for s in ODT_STOP_IDS]
DEP_DELAY_DIST_OUT = load('in/xtr/dep_delay_dist_out.pkl')
gtfs_st_df = pd.read_csv('in/raw/gtfs/stop_times.txt')
gtfs_st_df_out = gtfs_st_df[gtfs_st_df['stop_headsign'] == 'Illinois Center']
gtfs_st_df_out = gtfs_st_df_out.drop_duplicates(subset='stop_sequence')
gtfs_st_df_out = gtfs_st_df_out.sort_values(by='stop_sequence')
DIST_TRAVELED_OUT = gtfs_st_df_out['shape_dist_traveled'].tolist()
STOPS_OUTBOUND = gtfs_st_df_out['stop_id'].astype(str).tolist()

# INBOUND
TRIP_TIMES1_PARAMS = load('in/xtr/trip_time1_params.pkl')
TRIP_TIMES2_PARAMS = load('in/xtr/trip_time2_params.pkl')
TRIPS1_IN_INFO = load('in/xtr/trips1_info_inbound.pkl')
TRIPS2_IN_INFO = load('in/xtr/trips2_info_inbound.pkl')
DEP_DELAY1_DIST_IN = load('in/xtr/dep_delay1_dist_in.pkl')
DEP_DELAY2_DIST_IN = load('in/xtr/dep_delay2_dist_in.pkl')
TRIP_T1_DIST_IN = load('in/xtr/trip_t1_dist_in.pkl')
TRIP_T2_DIST_IN = load('in/xtr/trip_t2_dist_in.pkl')

LINK_TIMES_MEAN, LINK_TIMES_EXTREMES, LINK_TIMES_PARAMS = LINK_TIMES_INFO

SCALED_ARR_RATES = np.sum(ODT_RATES_SCALED, axis=-1)

TRIP_IDS_OUT, SCHED_DEP_OUT, BLOCK_IDS_OUT = [], [], []
for item in TRIPS_OUT_INFO:
    TRIP_IDS_OUT.append(item[0]), SCHED_DEP_OUT.append(item[1]), BLOCK_IDS_OUT.append(item[2])
trips_out = [(x, y, str(timedelta(seconds=y)), z, 0, w, v) for x, y, z, w, v in TRIPS_OUT_INFO]
trips_in1 = [(x, y, str(timedelta(seconds=y)), z, 1, w, v) for x, y, z, w, v in TRIPS1_IN_INFO]
trips_in2 = [(x, y, str(timedelta(seconds=y)), z, 2, w, v) for x, y, z, w, v in TRIPS2_IN_INFO]

trips_df = pd.DataFrame(trips_out + trips_in1 + trips_in2,
                        columns=['trip_id', 'schd_sec', 'schd_time', 'block_id', 'route_type', 'schedule', 'stops'])
sched_link_times = trips_df[trips_df['trip_id'] == TRIP_WITH_FULL_STOP_PATTERN]['schedule'].iloc[0]
SCHED_LINK_T = {}
for i in range(len(STOPS_OUTBOUND)-1):
    SCHED_LINK_T[STOPS_OUTBOUND[i] + '-' + STOPS_OUTBOUND[i+1]] = sched_link_times[i+1] - sched_link_times[i]
LINK_TIMES_PARAMS['3954-8613'] = [np.nan for i in range(TIME_NR_INTERVALS)]
LINK_TIMES_MEAN['3954-8613'] = [SCHED_LINK_T['3954-8613'] for i in range(TIME_NR_INTERVALS)]
# for ky in LINK_TIMES_MEAN:
#     print(f'link {ky} yields {LINK_TIMES_MEAN[ky]} and {SCHED_LINK_T[ky]} and {LINK_TIMES_EXTREMES[ky]}')
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
block_ids = trips_df['block_id'].unique().tolist()
BLOCK_TRIPS_INFO = []

for b in block_ids:
    block_df = trips_df[trips_df['block_id'] == b]
    trip_ids = block_df['trip_id'].tolist()
    lst_stops = block_df['stops'].tolist()
    lst_schedule = block_df['schedule'].tolist()
    route_types = block_df['route_type'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, route_types, lst_stops, lst_schedule))))
PAX_INIT_TIME = [0]
for s0, s1 in zip(STOPS_OUTBOUND, STOPS_OUTBOUND[1:]):
    ltimes = np.array(LINK_TIMES_MEAN[s0 + '-' + s1])
    ltime = ltimes[np.isfinite(ltimes)][0]
    PAX_INIT_TIME.append(ltime)
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_OUT[0] - ((SCHED_DEP_OUT[1] - SCHED_DEP_OUT[0]) / 2)

# trip id focused for results
ordered_trips_arr = np.array([TRIP_IDS_OUT])
sched_deps_arr = np.array([SCHED_DEP_OUT])
FOCUS_TRIPS = ordered_trips_arr[
    (sched_deps_arr <= FOCUS_END_TIME_SEC) & (sched_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
FOCUS_TRIPS_SCHED = sched_deps_arr[
    (sched_deps_arr <= FOCUS_END_TIME_SEC) & (sched_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
focus_trips_hw = [i - j for i, j in zip(FOCUS_TRIPS_SCHED[1:], FOCUS_TRIPS_SCHED[:-1])]
FOCUS_TRIPS_MEAN_HW = np.mean(focus_trips_hw)
FOCUS_TRIPS_HW_CV = round(np.std(focus_trips_hw) / np.mean(focus_trips_hw), 2)

# trip id focused for control
NO_CONTROL_TRIP_IDS = TRIP_IDS_OUT[:9] + TRIP_IDS_OUT[-11:]
NO_CONTROL_SCHED = SCHED_DEP_OUT[:9] + SCHED_DEP_OUT[-11:]
CONTROL_TRIP_IDS = TRIP_IDS_OUT[9:-11]
CONTROL_SCHEDULE = SCHED_DEP_OUT[9:-11]
CONTROL_HW = [t1 - t0 for t1, t0 in zip(CONTROL_SCHEDULE[1:], CONTROL_SCHEDULE[:-1])]
CONTROL_MEAN_HW = sum(CONTROL_HW) / len(CONTROL_HW)

BASE_HOLDING_TIME = 25
MIN_HW_THRESHOLD = 0.4
LIMIT_HOLDING = int(MIN_HW_THRESHOLD * CONTROL_MEAN_HW - MIN_HW_THRESHOLD * CONTROL_MEAN_HW % BASE_HOLDING_TIME)
N_ACTIONS_RL = int(LIMIT_HOLDING / BASE_HOLDING_TIME) + 2

# FOR UNIFORM CONDITIONS: TO USE - SET TIME-DEPENDENT TRAVEL TIME AND DEMAND TO FALSE
UNIFORM_INTERVAL = 1
SINGLE_LINK_TIMES_MEAN = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_MEAN.items()}
SINGLE_LINK_TIMES_PARAMS = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_PARAMS.items()}
SINGLE_LINK_TIMES_EXTREMES = {key: value[UNIFORM_INTERVAL] for (key, value) in LINK_TIMES_EXTREMES.items()}
