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
dir_figs = 'fig/'
dir_csv = 'txt/'
dir_var = 'var/'

# INPUT FILES

path_trips_gtfs = path_to_ins + dir_raw + 'gtfs/trips.txt'
path_stop_times = path_to_ins + dir_raw + 'route20_stop_time_merged.csv'
path_stops_loc = path_to_ins + dir_raw + 'gtfs/stops.txt'
path_od = path_to_ins + dir_raw + 'odt_for_opt.csv'
path_dispatching_times = path_to_ins + dir_raw + 'dispatching_time.csv'
path_ordered_dispatching = path_to_ins + dir_raw + 'inbound_ordered_dispatching.csv'

# EXTRACT NETWORK PARAMS

path_route_stops = path_to_ins + 'xtr/' + dir_project + 'route_stops' + '.pkl'
path_link_times_mean = path_to_ins + 'xtr/' + dir_project + 'link_times_mean' + '.pkl'
path_link_times_sd = path_to_ins + 'xtr/' + dir_project + 'link_times_sd' + '.pkl'
path_link_dpoints = path_to_ins + 'xtr/' + dir_project + 'link_dpoints' + '.pkl'
path_ordered_trips = path_to_ins + 'xtr/' + dir_project + 'link_dpoints' + '.pkl'
path_arr_rates = path_to_ins + 'xtr/' + dir_project + 'arr_rates' + '.pkl'
path_alight_fractions = path_to_ins + 'xtr/' + dir_project + 'alight_fractions' + '.pkl'
path_departure_times_xtr = path_to_ins + 'xtr/' + dir_project + 'departure_times' + '.pkl'
path_alight_rates = path_to_ins + 'xtr/' + dir_project + 'alight_rates' + '.pkl'
path_dep_volume = path_to_ins + 'xtr/' + dir_project + 'dep_vol' + '.pkl'
path_odt_xtr = path_to_ins + 'xtr/' + dir_project + 'odt' + '.pkl'

# VISUALIZE NETWORK PARAMS

path_sorted_daily_trips = path_to_ins + dir_vis + 'trips_'
path_input_cv_link_times = path_to_ins + dir_vis + 'cv_link_times.png'
path_historical_headway = path_to_ins + dir_vis + 'historical_headway.png'
path_input_load_profile = path_to_ins + dir_vis + 'input_load_profile.png'
path_input_link_times = path_to_ins + dir_vis + 'travel_time_distribution.csv'
path_input_link_times_fig = path_to_ins + dir_vis + 'input_link_times.png'
path_stop_pattern = path_to_ins + dir_vis + 'stop_pattern.csv'
path_odt_fig = path_to_ins + dir_vis + 'od0.png'
