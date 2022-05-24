# from pre_process import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pre_process import extract_apc_counts, bi_proportional_fitting, get_trip_times, get_load_profile
from file_paths import *
from constants import *
from post_process import save, load
from datetime import timedelta


def extract_params(demand=False, validation=False):
    if demand:
        stops_outbound = load(path_route_stops)
        odt_stops = np.load('in/xtr/rt_20_odt_stops.npy')
        # comes from project with dingyi data
        odt_pred = np.load('in/xtr/rt_20_odt_rates_30.npy')
        # comes from project with dingyi data

        nr_intervals = 24 / (ODT_INTERVAL_LEN_MIN / 60)
        apc_on_rates, apc_off_rates = extract_apc_counts(nr_intervals, odt_stops, path_stop_times, ODT_INTERVAL_LEN_MIN,
                                                         DATES)
        stops_lst = list(odt_stops)

        # DISCOVERED IN DINGYI'S OD MATRIX TIME SHIFT
        shifted_odt = np.concatenate((odt_pred[-6:], odt_pred[:-6]), axis=0)
        scaled_odt = np.concatenate((odt_pred[-6:], odt_pred[:-6]), axis=0)

        for i in range(shifted_odt.shape[0]):
            print(f'interval {i}')
            scaled_odt[i] = bi_proportional_fitting(shifted_odt[i], apc_on_rates[i], apc_off_rates[i])

        np.save('in/xtr/rt_20_odt_rates_30_scaled.npy', scaled_odt)

        # if wanted for comparison
        idx_stops_out = [stops_lst.index(int(s)) for s in stops_outbound]
        out_on_counts = apc_on_rates[:, idx_stops_out]
        out_on_tot_count = np.nansum(out_on_counts, axis=-1)

        arr_rates_shifted = np.nansum(shifted_odt, axis=-1)
        out_arr_rates_shifted = arr_rates_shifted[:, idx_stops_out]
        out_arr_tot_shifted = np.sum(out_arr_rates_shifted, axis=-1)

        scaled_arr_rates = np.sum(scaled_odt, axis=-1)
        scaled_out_arr_rates = scaled_arr_rates[:, idx_stops_out]
        scaled_out_tot = np.sum(scaled_out_arr_rates, axis=-1)

        x = np.arange(out_on_tot_count.shape[0])
        plt.plot(x, scaled_out_tot, label='odt scaled')
        plt.plot(x, out_arr_tot_shifted, label='odt')
        plt.plot(x, out_on_tot_count, label='apc')
        plt.xticks(np.arange(0, out_on_tot_count.shape[0], 2), np.arange(int(out_on_tot_count.shape[0] / 2)))
        plt.xlabel('hour of day')
        plt.ylabel('arrival rate (1/h)')
        plt.yticks(np.arange(0, 1200, 200))
        plt.legend()
        # plt.show()
        plt.close()

    if validation:
        stops = load(path_route_stops)
        trips_outbound_info = load('in/xtr/trips_outbound_info.pkl')
        scheduled_dep_in, ordered_trips_in = [], []
        for t in trips_outbound_info:
            scheduled_dep_in.append(t[1]), ordered_trips_in.append(t[0])
        ordered_trips_in = np.array(ordered_trips_in)
        schedule_arr = np.array(scheduled_dep_in)
        focus_trips = ordered_trips_in[
            (schedule_arr <= FOCUS_END_TIME_SEC) & (schedule_arr >= FOCUS_START_TIME_SEC)].tolist()
        trip_times = get_trip_times(path_avl, focus_trips, DATES, stops)
        save('in/xtr/trip_t_outbound.pkl', trip_times)
        load_profile, ons, offs = get_load_profile(path_stop_times, focus_trips, stops)
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.plot(load_profile)
        x = np.arange(len(load_profile))
        w = 0.5
        ax1.bar(x, ons, w)
        ax1.bar(x + w, offs, w)
        plt.savefig('in/vis/pax_profile_observed.png')
        plt.close()
        save('in/xtr/load_profile.pkl', load_profile)
    return


# extract_outbound_params(path_stop_times, START_TIME_SEC, END_TIME_SEC, TIME_NR_INTERVALS,
#                         TIME_START_INTERVAL, TIME_INTERVAL_LENGTH_MINS, DATES,
#                         TRIP_WITH_FULL_STOP_PATTERN, path_avl, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL)
# extract_inbound_params(
#     path_stop_times, START_TIME_SEC, END_TIME_SEC, DATES, TRIP_TIME_NR_INTERVALS, TRIP_TIME_START_INTERVAL,
#     TRIP_TIME_INTERVAL_LENGTH_MINS, path_avl, DELAY_INTERVAL_LENGTH_MINS, DELAY_START_INTERVAL)
# extract_params(validation=True)

# OUTBOUND
STOPS_OUTBOUND = load(path_route_stops)
LINK_TIMES_INFO = load(path_link_times_mean)
TRIPS_OUT_INFO = load('in/xtr/trips_outbound_info.pkl')
ODT_RATES_SCALED = np.load('in/xtr/rt_20_odt_rates_30_scaled.npy')
ODT_STOP_IDS = list(np.load('in/xtr/rt_20_odt_stops.npy'))
ODT_STOP_IDS = [str(int(s)) for s in ODT_STOP_IDS]
DEP_DELAY_DIST_OUT = load('in/xtr/dep_delay_dist_out.pkl')

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
trips_df['block_id'] = trips_df['block_id'].astype(str).str[6:].astype(int)
trips_df = trips_df.sort_values(by=['block_id', 'schd_sec'])
# trips_df.to_csv('in/vis/block_info.csv', index=False)
block_ids = trips_df['block_id'].unique().tolist()
BLOCK_TRIPS_INFO = []
BLOCK_DICT = {}
# avl_df = pd.read_csv('in/raw/rt20_avl.csv')
for b in block_ids:
    block_df = trips_df[trips_df['block_id'] == b]
    trip_ids = block_df['trip_id'].tolist()
    sched_deps = block_df['schd_sec'].tolist()
    lst_stops = block_df['stops'].tolist()
    lst_schedule = block_df['schedule'].tolist()
    BLOCK_DICT[b] = trip_ids
    route_types = block_df['route_type'].tolist()
    BLOCK_TRIPS_INFO.append((b, list(zip(trip_ids, sched_deps, route_types, lst_stops, lst_schedule))))

PAX_INIT_TIME = [0]
for s0, s1 in zip(STOPS_OUTBOUND, STOPS_OUTBOUND[1:]):
    ltimes = np.array(LINK_TIMES_MEAN[s0 + '-' + s1])
    ltime = ltimes[np.isfinite(ltimes)][0]
    PAX_INIT_TIME.append(ltime)
PAX_INIT_TIME = np.array(PAX_INIT_TIME).cumsum()
PAX_INIT_TIME += SCHED_DEP_OUT[0] - ((SCHED_DEP_OUT[1] - SCHED_DEP_OUT[0]) / 2)

# trip id focused for results
ordered_trips_arr = np.array([TRIP_IDS_OUT])
scheduled_deps_arr = np.array([SCHED_DEP_OUT])
FOCUS_TRIPS = ordered_trips_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
FOCUS_TRIPS_SCHED = scheduled_deps_arr[
    (scheduled_deps_arr <= FOCUS_END_TIME_SEC) & (scheduled_deps_arr >= FOCUS_START_TIME_SEC)].tolist()
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
