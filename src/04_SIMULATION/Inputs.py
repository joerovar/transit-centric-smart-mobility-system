import numpy as np
import pandas as pd
from Input_Processor import extract_outbound_params, extract_inbound_params, extract_demand
# from File_Paths import *
from File_Paths import dir_route
from datetime import datetime
import math
from Output_Processor import load
from datetime import timedelta

# SIMULATION

START_TIME = datetime.strptime('05:00:00', "%H:%M:%S")
END_TIME = datetime.strptime('10:00:00', "%H:%M:%S")
FOCUS_START_TIME = datetime.strptime('07:28:00', "%H:%M:%S")
FOCUS_END_TIME = datetime.strptime('08:15:00', "%H:%M:%S")

START_TIME_SEC = (START_TIME - datetime(1900, 1, 1)).total_seconds()
END_TIME_SEC = (END_TIME - datetime(1900, 1, 1)).total_seconds()
TOTAL_MIN = (END_TIME - START_TIME).total_seconds() / 60
FOCUS_START_TIME_SEC = (FOCUS_START_TIME - datetime(1900, 1, 1)).total_seconds()
FOCUS_END_TIME_SEC = (FOCUS_END_TIME - datetime(1900, 1, 1)).total_seconds()
DELAY_INTERVAL_LENGTH_MINS = 60
DELAY_START_INTERVAL = int(START_TIME_SEC / (60 * DELAY_INTERVAL_LENGTH_MINS))
TIME_INTERVAL_LENGTH_MINS = 30
TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TIME_INTERVAL_LENGTH_MINS))
TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TIME_INTERVAL_LENGTH_MINS))
DATES = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06',
         '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12', '2019-09-13',
         '2019-09-16', '2019-09-17', '2019-09-18', '2019-09-19', '2019-09-20',
         '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27']
# INBOUND TO OUTBOUND LAYOVER TIME
MIN_LAYOVER_T = 50
ERR_LAYOVER_TIME = 20
# INBOUND TIME DEPENDENT TRIP TIME DISTRIBUTION
TRIP_TIME_INTERVAL_LENGTH_MINS = 60
TRIP_TIME_START_INTERVAL = int(START_TIME_SEC / (60 * TRIP_TIME_INTERVAL_LENGTH_MINS))
TRIP_TIME_NR_INTERVALS = int(math.ceil(TOTAL_MIN / TRIP_TIME_INTERVAL_LENGTH_MINS))
# TRAVEL, DWELL TIME AND DEPARTURE DELAY DISTRIBUTION
ACC_DEC_TIME = 4.5
BOARDING_TIME = 2.3
ALIGHTING_TIME = 1.2
DWELL_TIME_ERROR = 3.0
EXTREME_TT_BOUND = 1.0
BOOST_SCHED_RUN_T = 1.3 # FOR THOSE PROBLEMATIC LINKS (FIRST AND LAST) INCREASE SCHED RUN TIME BY 30 PCT
HIGH_CAPACITY = 80
LOW_CAPACITY = 53
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

CONTROLLED_STOPS = ['386', '409', '423', '16049', '3954']
CONTROLLED_STOPS_ALTERNATIVE = ['386', '409', '428', '3954']
N_STATE_PARAMS_RL = 6
[IDX_RT_PROGRESS, IDX_LOAD_RL, IDX_FW_H, IDX_BW_H, IDX_PAX_AT_STOP, IDX_PREV_FW_H] = [i for i in
                                                                                      range(N_STATE_PARAMS_RL)]
SKIP_ACTION = 0
ESTIMATED_PAX = False
WEIGHT_RIDE_T = 0.0
TT_FACTOR = 1.0
HOLD_ADJ_FACTOR = 0.0
FULL_PATTERN_HEADSIGN = 'Illinois Center'
RT_NR = 20
OB_DIRECTION = 'East'
IB_DIRECTION = 'West'

# EXTRACT FUNCTIONS
# extract_demand(ODT_INTERVAL_LEN_MIN, DATES)
# extract_outbound_params(START_TIME_SEC, END_TIME_SEC, TIME_NR_INTERVALS, TIME_START_INTERVAL, TIME_INTERVAL_LENGTH_MINS,
#                         DATES, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL, FULL_PATTERN_HEADSIGN, RT_NR,
#                         OB_DIRECTION)
# extract_inbound_params(START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
#                        TRIP_TIME_INTERVAL_LENGTH_MINS, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL, RT_NR,
#                        IB_DIRECTION)

# OUTBOUND
LINK_TIMES_INFO = load(dir_route + 'link_times_info.pkl')
TRIPS_OUT_INFO = load(dir_route + 'trips_out_info.pkl')
ODT_RATES_SCALED = np.load(dir_route + 'odt_rates_30_scaled.npy')
ODT_STOP_IDS = list(np.load(dir_route + 'odt_stops.npy'))
ODT_STOP_IDS = [str(int(s)) for s in ODT_STOP_IDS]
DEP_DELAY_DIST_OUT = load(dir_route + 'dep_delay_dist_out.pkl') # empirical delay data , including negative
STOPS_OUT_FULL_PATT = load(dir_route + 'stops_out_full_patt.pkl')
STOPS_OUT_ALL = load(dir_route + 'stops_out_all.pkl')
STOPS_OUT_INFO = load(dir_route + 'stops_out_info.pkl')

# INBOUND
TRIPS_IN_INFO = load(dir_route + 'trips_in_info.pkl')
RUN_T_DIST_IN = load(dir_route + 'run_times_in.pkl')
DELAY_DIST_IN = load(dir_route + 'delay_in.pkl')

LINK_TIMES_MEAN, LINK_TIMES_EXTREMES, LINK_TIMES_PARAMS = LINK_TIMES_INFO
SCALED_ARR_RATES = np.sum(ODT_RATES_SCALED, axis=-1)
TRIP_IDS_OUT, SCHED_DEP_OUT, BLOCK_IDS_OUT = [], [], []
for item in TRIPS_OUT_INFO:
    TRIP_IDS_OUT.append(item[0]), SCHED_DEP_OUT.append(item[1]), BLOCK_IDS_OUT.append(item[2])
trips_out = [(x, y, str(timedelta(seconds=y)), z, 0, w, v, u) for x, y, z, w, v, u in TRIPS_OUT_INFO]
trips_in = [(x, y, str(timedelta(seconds=y)), z, 1, w, v, u) for x, y, z, w, v, u in TRIPS_IN_INFO]

trips_df = pd.DataFrame(trips_out + trips_in, columns=['trip_id', 'schd_sec', 'schd_time',
                                                       'block_id', 'route_type', 'schedule', 'stops', 'dist_traveled'])
outbound_trips_df = trips_df[trips_df['route_type']==0]
outbound_trips_df = outbound_trips_df.sort_values(by='schd_sec')
trip_ids = outbound_trips_df['trip_id'].tolist()
trip_schedules = outbound_trips_df['schedule'].tolist()
trip_dist_traveled = outbound_trips_df['dist_traveled'].tolist()
scheduled_trajectories_out = []
for i in range(len(trip_ids)):
    for j in range(len(trip_schedules[i])):
        scheduled_trajectories_out.append([trip_ids[i], trip_schedules[i][j], trip_dist_traveled[i][j]])

df_sched_t_rep = pd.DataFrame(scheduled_trajectories_out, columns=['trip_id', 'schd_sec', 'dist_traveled'])

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
    lst_dist_traveled = block_df['dist_traveled'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, route_types, lst_stops, lst_schedule, lst_dist_traveled))))
PAX_INIT_TIME = [0]
for s0, s1 in zip(STOPS_OUT_FULL_PATT, STOPS_OUT_FULL_PATT[1:]):
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

# TERMINAL DISPATCH PARAMS
MRH_STOPS = [STOPS_OUT_FULL_PATT[27], STOPS_OUT_FULL_PATT[43]]

# p = 0.15
# nr_iters = 30
# cancelled = []
# for i in range(nr_iters):
#     tmp_cancelled = 0
#     trip_ids_cancelled = []
#     for j in range(len(BLOCK_TRIPS_INFO)):
#         if np.random.uniform(0, 1) < p:
#             tmp_cancelled += len([b for b in BLOCK_TRIPS_INFO[j][1] if b[1]==0])
#             trip_ids_cancelled += [b[0] for b in BLOCK_TRIPS_INFO[j][1] if b[1]==0]
#     remaining_sched = [SCHED_DEP_OUT[k] for k in range(len(SCHED_DEP_OUT)) if TRIP_IDS_OUT[k] not in trip_ids_cancelled]
#     remaining_hw = [round((remaining_sched[k+1]-remaining_sched[k])/60,2) for k in range(len(remaining_sched)-1)]
#     true_hw = [round((SCHED_DEP_OUT[k+1]-SCHED_DEP_OUT[k])/60,2) for k in range(len(SCHED_DEP_OUT)-1)]
#     cancelled.append(tmp_cancelled)
# total_trips = len(TRIP_IDS_OUT)
# print(np.mean(cancelled)/total_trips)
# print(np.mean(cancelled))
# print(np.std(cancelled))