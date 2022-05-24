import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import DATES, START_TIME_SEC, END_TIME_SEC
from input import STOPS_OUTBOUND
from pre_process import get_interval, remove_outliers
from post_process import plot_calib_hist, validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound

sim_df = pd.read_pickle('out/NC/0524-153428-trip_record_outbound.pkl')
avl_df = pd.read_csv('in/raw/rt20_avl.csv')
sim_df_in = pd.read_pickle('out/NC/0524-153428-trip_record_inbound.pkl')

# trip ids between focus start-end
# avl
focus_avl_df = avl_df[avl_df['stop_sequence'] == 1]
focus_avl_df = focus_avl_df[focus_avl_df['stop_id'] == int(STOPS_OUTBOUND[0])]
focus_avl_df = focus_avl_df[focus_avl_df['schd_sec'] < END_TIME_SEC]
focus_avl_df = focus_avl_df[focus_avl_df['schd_sec'] >= START_TIME_SEC]
focus_avl_df = focus_avl_df.sort_values(by='schd_sec')
avl_focus_trips = focus_avl_df['trip_id'].unique().tolist()

focus_sim_df = sim_df[sim_df['stop_sequence'] == 1]
focus_sim_df = focus_sim_df[focus_sim_df['stop_id'] == STOPS_OUTBOUND[0]]
focus_sim_df = focus_sim_df[focus_sim_df['schd_sec'] < END_TIME_SEC]
focus_sim_df = focus_sim_df[focus_sim_df['schd_sec'] >= START_TIME_SEC]
focus_sim_df = focus_sim_df.sort_values(by='schd_sec')
sim_focus_trips = focus_sim_df['trip_id'].unique().tolist()

interval_mins = 60
interval0 = get_interval(START_TIME_SEC, interval_mins)
interval1 = get_interval(END_TIME_SEC, interval_mins)
trip_t_avl = [[] for _ in range(interval0, interval1)]
dwell_t_avl = [[] for _ in range(interval0, interval1)]
for d in DATES:
    date_df = avl_df[avl_df['avl_arr_time'].astype(str).str[:10] == d]
    for trip in avl_focus_trips:
        trip_df = date_df[date_df['trip_id'] == trip]
        t0 = trip_df[trip_df['stop_sequence'] == 1]
        t1 = trip_df[trip_df['stop_sequence'] == 67]
        if not t0.empty and not t1.empty:
            t0 = t0.iloc[0]
            t1 = t1.iloc[0]
            interval = get_interval(t0['schd_sec'], interval_mins)
            dep_t = t0['avl_dep_sec'].astype(int)
            arr_t = t1['avl_arr_sec'].astype(int)
            trip_t_avl[interval-interval0].append(arr_t-dep_t)

            mid_route_df = trip_df[(trip_df['stop_sequence']!=1) & (trip_df['stop_sequence'] != 67)]
            mid_route_df = mid_route_df.drop_duplicates(subset='stop_sequence', keep='first')
            if 67-4 <= mid_route_df.shape[0] <= 67-2:
                mid_route_df['dwell_t'] = mid_route_df['avl_dep_sec'] - mid_route_df['avl_arr_sec']
                dwell_t_avl[interval-interval0].append(mid_route_df['dwell_t'].sum())
            else:
                lst_stops = mid_route_df['stop_sequence'].tolist()
for i in range(interval1-interval0):
    if trip_t_avl[i]:
        trip_t_avl[i] = remove_outliers(np.array(trip_t_avl[i])).tolist()
    if dwell_t_avl[i]:
        dwell_t_avl[i] = remove_outliers(np.array(dwell_t_avl[i])).tolist()

trip_t_sim = [[] for _ in range(interval0, interval1)]
dwell_t_sim = [[] for _ in range(interval0, interval1)]
nr_replications = sim_df['replication'].max()
for i in range(nr_replications):
    rep_df = sim_df[sim_df['replication'] == i+1]
    for trip in sim_focus_trips:
        trip_df = rep_df[rep_df['trip_id'] == trip]
        t0 = trip_df[trip_df['stop_sequence'] == 1]
        t1 = trip_df[trip_df['stop_sequence'] == 67]
        if not t0.empty and not t1.empty:
            t0 = t0.iloc[0]
            t1 = t1.iloc[0]
            interval = get_interval(t0['schd_sec'], interval_mins)
            dep_t = t0['dep_sec'].astype(int)
            arr_t = t1['arr_sec'].astype(int)
            trip_t_sim[interval - interval0].append(arr_t - dep_t)
            mid_route_df = trip_df[(trip_df['stop_sequence']!=1) & (trip_df['stop_sequence'] != 67)]
            if mid_route_df.shape[0] == 67-2:
                mid_route_df['dwell_t'] = mid_route_df['dep_sec'] - mid_route_df['arr_sec']
                dwell_t_sim[interval-interval0].append(mid_route_df['dwell_t'].sum())

validate_delay_outbound(avl_df, sim_df, START_TIME_SEC, END_TIME_SEC)
validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
validate_cv_hw_outbound(avl_df, sim_df, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUTBOUND, DATES)
plot_calib_hist(trip_t_avl, trip_t_sim, 5, 'out/compare/validate/trip_t_dist_out.png', 'total trip time (seconds)')
plot_calib_hist(dwell_t_avl, dwell_t_sim, 5, 'out/compare/validate/dwell_t_dist_out.png', 'dwell time (seconds)')
