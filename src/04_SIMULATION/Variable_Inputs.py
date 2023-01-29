from typing import KeysView
import numpy as np
import pandas as pd
from Input_Processor import extract_outbound_params, extract_inbound_params, extract_demand
from ins.Fixed_Inputs_81 import *
# from ins.Fixed_Inputs_20 import *
from Output_Processor import load
from datetime import timedelta
import matplotlib.pyplot as plt

def time_string(secs):
    return str(timedelta(seconds=round(secs)))

# EXTRACT FUNCTIONS
# extract_outbound_params(START_TIME_SEC, END_TIME_SEC, TIME_NR_INTERVALS, TIME_START_INTERVAL, TIME_BIN_MINS,
#                         DATES, DELAY_BIN_MINS, DELAY_START_INTERVAL, FULL_PATTERN_HEADSIGN, RT_NR, OB_DIRECTION,
#                         min_speed=3, max_speed=25)
# extract_inbound_params(START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
#                        TRIP_TIME_BIN_MINS, DELAY_BIN_MINS, DELAY_START_INTERVAL, RT_NR, IB_DIRECTION)
# extract_demand(ODT_BIN_MINS, DATES, 'avl.csv')


# OUTBOUND
LINK_TIMES_INFO = load(DIR_ROUTE + 'link_times_info.pkl')
TRIPS_OUT_INFO = load(DIR_ROUTE + 'trips_out_info.pkl')
ODT_FLOWS = np.load(DIR_ROUTE + 'odt_flows_30_scaled.npy')
ODT_STOP_IDS = list(np.load(DIR_ROUTE + 'odt_stops.npy'))
DELAY_DISTR_OUT = load(DIR_ROUTE + 'delay_distr_out.pkl')  # empirical delay data , including negative
STOPS_OUT_FULL_PATT = load(DIR_ROUTE + 'stops_out_full_patt.pkl')
STOPS_OUT_ALL = load(DIR_ROUTE + 'stops_out_all.pkl')
STOPS_OUT_INFO = load(DIR_ROUTE + 'stops_out_info.pkl')
STOPS_OUT_NAMES = pd.read_csv(DIR_ROUTE + 'gtfs_stops_route.txt')['short_name'].tolist()
KEY_STOPS_IDX = [STOPS_OUT_NAMES.index(s) for s in ['TRANSIT CENTER','CICERO', 
                                                    'PULASKI', 'KIMBALL (BROWN LINE)', 'WESTERN', 
                                                    'RAVENSWOOD', 'BROADWAY (RED LINE)', 'MARINE DRIVE']]                 
# print(STOPS_OUT_FULL_PATT[KEY_STOPS_IDX[3]])
# INBOUND
TRIPS_IN_INFO = load(DIR_ROUTE + 'trips_in_info.pkl')
RUN_T_DISTR_IN = load(DIR_ROUTE + 'run_t_distr_in.pkl')
DELAY_DISTR_IN = load(DIR_ROUTE + 'delay_distr_in.pkl')
STOPS_IN_FULL_PATT = TRIPS_IN_INFO[0][4]
ARR_RATES = np.sum(ODT_FLOWS, axis=-1)
LINK_TIMES_MEAN, LINK_TIMES_EXTREMES, LINK_TIMES_PARAMS = LINK_TIMES_INFO

TRIP_IDS_OUT, SCHED_DEP_OUT, BLOCK_IDS_OUT = [], [], []
for item in TRIPS_OUT_INFO:
    TRIP_IDS_OUT.append(item[0]), SCHED_DEP_OUT.append(item[1]), BLOCK_IDS_OUT.append(item[2])
trips_out = [(x, y, str(timedelta(seconds=y)), z, 0, w, v, u) for x, y, z, w, v, u in TRIPS_OUT_INFO]
trips_in = [(x, y, str(timedelta(seconds=y)), z, 1, w, v, u) for x, y, z, w, v, u in TRIPS_IN_INFO]
trips_df = pd.DataFrame(trips_out + trips_in, columns=['trip_id', 'schd_sec', 'schd_time',
                                                       'block_id', 'route_type', 'schedule', 'stops', 'dist_traveled'])

trips_out_df = trips_df[trips_df['route_type'] == 0]
trips_out_df = trips_out_df.sort_values(by='schd_sec')

trip_ids_out = trips_out_df['trip_id'].tolist()
trip_schedules_out = trips_out_df['schedule'].tolist()
trip_dist_traveled_out = trips_out_df['dist_traveled'].tolist()
scheduled_trajectories_out = []
for i in range(len(trip_ids_out)):
    for j in range(len(trip_schedules_out[i])):
        scheduled_trajectories_out.append([trip_ids_out[i], trip_schedules_out[i][j], trip_dist_traveled_out[i][j]])

df_sched_t_rep = pd.DataFrame(scheduled_trajectories_out, columns=['trip_id', 'schd_sec', 'dist_traveled'])
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
BLOCK_IDS = trips_df['block_id'].unique().tolist()

BLOCK_TRIPS_INFO = []

for b in BLOCK_IDS:
    block_df = trips_df[trips_df['block_id'] == b].copy()
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
FOCUS_START_TIME_SEC = (FOCUS_START_TIME - datetime(1900, 1, 1)).total_seconds()
FOCUS_END_TIME_SEC = (FOCUS_END_TIME - datetime(1900, 1, 1)).total_seconds()
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

LIMIT_HOLDING = int(MIN_HW_THRESHOLD * CONTROL_MEAN_HW - MIN_HW_THRESHOLD * CONTROL_MEAN_HW % BASE_HOLDING_TIME)
N_ACTIONS_RL = int(LIMIT_HOLDING / BASE_HOLDING_TIME) + 2
