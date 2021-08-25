import time

path_stops_loc = 'in/gtfs/stops.txt'
path_to_outs = 'out/'
ext_var = '.pkl'
ext_fig = '.png'
ext_csv = '.csv'
tstamp_save = time.strftime("%m%d-%H%M")
tstamp_load = tstamp_save

# SAVE

path_tr_save = 'out/trajectories_' + tstamp_save + '.pkl'
path_hw_save = 'out/headway_' + tstamp_save + '.pkl'
path_wt_save = 'out/stop_wait_times_' + tstamp_save + '.pkl'
path_bd_save = 'out/boardings_' + tstamp_save + '.pkl'
path_db_save = 'out/denied_boardings_' + tstamp_save + '.pkl'
path_wtc_save = 'out/stop_wait_times_from_h_' + tstamp_save + '.pkl'

# LOAD

path_tr_load = 'out/trajectories_' + tstamp_load + '.pkl'
path_hw_load = 'out/headway_' + tstamp_load + '.pkl'
path_wt_load = 'out/stop_wait_times_' + tstamp_load + '.pkl'
path_bd_load = 'out/boardings_' + tstamp_load + '.pkl'
path_db_load = 'out/denied_boardings_' + tstamp_load + '.pkl'
path_wtc_load = 'out/stop_wait_times_from_h_' + tstamp_load + '.pkl'
path_lt = 'out/link_times_' + tstamp_load + '.csv'
path_wt = 'out/stop_wait_times_' + tstamp_load + '.csv'
path_hw_fig = 'out/headway_' + tstamp_load + '.png'
path_tr_csv = 'out/trajectories_' + tstamp_load + '.csv'
path_tr_fig = 'out/trajectories_' + tstamp_load + '.png'
path_wtc_fig = 'out/wait_time_compare_' + tstamp_load + '.png'
path_bd_fig = 'out/boardings_' + tstamp_load + '.png'
path_db_fig = 'out/denied_boardings_' + tstamp_load + '.png'
