import time

path_to_ins = 'in/'
path_to_outs = 'out/'
ext_var = '.pkl'
ext_fig = '.png'
ext_csv = '.csv'
tstamp_save = time.strftime("%m%d-%H%M")
tstamp_load = tstamp_save

# INPUT FILES
path_trips_gtfs = path_to_ins + 'gtfs/trips.txt'
path_stop_times = path_to_ins + 'route20_stop_time.csv'
path_stops_loc = path_to_ins + 'gtfs/stops.txt'
path_od = path_to_ins + 'odt_for_opt.csv'
path_dispatching_times = path_to_ins + 'dispatching_time.csv'

# WRITE / PLOT INPUT

path_historical_headway = path_to_outs + 'historic_headway_!.png'
path_input_boardings = path_to_outs + 'predicted_boardings.png'
path_link_times = path_to_outs + 'travel_time_distribution.csv'

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
path_bd_fig = path_to_outs + 'boardings_' + tstamp_load + '.png'
path_db_fig = path_to_outs + 'denied_boardings_' + tstamp_load + '.png'
