import time

path_to_ins = 'in/'
path_to_outs = 'out/'
ext_var = '.pkl'
ext_fig = '.png'
ext_csv = '.csv'
tstamp_save = time.strftime("%m%d-%H%M")
tstamp_load = tstamp_save
dir_project = 'rt_20-2019-09/'
dir_raw = 'raw/'
dir_vis = 'vis/'
# INPUT FILES

path_trips_gtfs = path_to_ins + dir_raw + 'gtfs/trips.txt'
path_stop_times = path_to_ins + dir_raw + 'route20_stop_time.csv'
path_stops_loc = path_to_ins + dir_raw + 'gtfs/stops.txt'
path_od = path_to_ins + dir_raw + 'odt_for_opt.csv'
path_dispatching_times = path_to_ins + dir_raw + 'dispatching_time.csv'
path_ordered_dispatching = path_to_ins + dir_raw + 'ordered_dispatching.csv'

# EXTRACT NETWORK PARAMS

path_route_stops = path_to_ins + 'xtr/' + dir_project + 'route_stops' + '.pkl'
path_link_times_mean = path_to_ins + 'xtr/' + dir_project + 'link_times_mean' + '.pkl'
path_link_times_sd = path_to_ins + 'xtr/' + dir_project + 'link_times_sd' + '.pkl'
path_link_dpoints = path_to_ins + 'xtr/' + dir_project + 'link_dpoints' + '.pkl'
path_ordered_trips = path_to_ins + 'xtr/' + dir_project + 'link_dpoints' + '.pkl'
path_arr_rates = path_to_ins + 'xtr/' + dir_project + 'arr_rates' + '.pkl'
path_alight_fractions = path_to_ins + 'xtr/' + dir_project + 'alight_fractions' + '.pkl'
path_departure_times_xtr = path_to_ins + 'xtr/' + dir_project + 'departure_times' + '.pkl'

# VISUALIZE NETWORK PARAMS

path_sorted_daily_trips = path_to_ins + dir_vis + 'trips_'
path_input_cv_link_times = path_to_ins + dir_vis + 'cv_link_times.png'
path_historical_headway = path_to_ins + dir_vis + 'historic_headway_!.png'
path_input_boardings = path_to_ins + dir_vis + 'predicted_boardings.png'
path_input_link_times = path_to_ins + dir_vis + 'travel_time_distribution.csv'
path_stop_pattern = path_to_ins + dir_vis + 'stop_pattern.csv'

# SAVE OUTPUT

path_tr_save = path_to_outs + 'trajectories_' + tstamp_save + '.pkl'
path_hw_save = path_to_outs + 'headway_' + tstamp_save + '.pkl'
path_wt_save = path_to_outs + 'stop_wait_times_' + tstamp_save + '.pkl'
path_bd_save = path_to_outs + 'boardings_' + tstamp_save + '.pkl'
path_db_save = path_to_outs + 'denied_boardings_' + tstamp_save + '.pkl'
path_wtc_save = path_to_outs + 'stop_wait_times_from_h_' + tstamp_save + '.pkl'

# LOAD

path_tr_load = path_to_outs + 'trajectories_' + tstamp_load + '.pkl'
path_hw_load = path_to_outs + 'headway_' + tstamp_load + '.pkl'
path_wt_load = path_to_outs + 'stop_wait_times_' + tstamp_load + '.pkl'
path_bd_load = path_to_outs + 'boardings_' + tstamp_load + '.pkl'
path_db_load = path_to_outs + 'denied_boardings_' + tstamp_load + '.pkl'
path_wtc_load = path_to_outs + 'stop_wait_times_from_h_' + tstamp_load + '.pkl'

# WRITE / PLOT OUTPUT

path_lt = path_to_outs + 'link_times_' + tstamp_load + '.csv'
path_wt = path_to_outs + 'stop_wait_times_' + tstamp_load + '.csv'
path_hw_fig = path_to_outs + 'headway_' + tstamp_load + '.png'
path_tr_csv = path_to_outs + 'trajectories_' + tstamp_load + '.csv'
path_tr_fig = path_to_outs + 'trajectories_' + tstamp_load + '.png'
path_wtc_fig = path_to_outs + 'wait_time_compare_' + tstamp_load + '.png'
path_wt_fig = path_to_outs + 'stop_wait_times_' + tstamp_load + '.png'
path_bd_fig = path_to_outs + 'boardings_' + tstamp_load + '.png'
path_db_fig = path_to_outs + 'denied_boardings_' + tstamp_load + '.png'
