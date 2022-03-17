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
path_extra_stop_times = path_to_ins + dir_raw + 'rt20_extra_stop_times.csv'
path_stops_loc = path_to_ins + dir_raw + 'gtfs/stops.txt'
path_od = path_to_ins + dir_raw + 'odt_for_opt.csv'
path_dispatching_times = path_to_ins + dir_raw + 'dispatching_time.csv'
path_ordered_dispatching = path_to_ins + dir_raw + 'inbound_ordered_dispatching.csv'

# EXTRACT NETWORK PARAMS

path_route_stops = path_to_ins + 'xtr/' + dir_project + 'route_stops' + '.pkl'
path_link_times_mean = path_to_ins + 'xtr/' + dir_project + 'link_times_info' + '.pkl'
path_ordered_trips = path_to_ins + 'xtr/' + dir_project + 'ordered_trips' + '.pkl'
path_arr_rates = path_to_ins + 'xtr/' + dir_project + 'arr_rates' + '.pkl'
path_alight_fractions = path_to_ins + 'xtr/' + dir_project + 'alight_fractions' + '.pkl'
path_departure_times_xtr = path_to_ins + 'xtr/' + dir_project + 'departure_times_inbound' + '.pkl'
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

# 70 REPLICATIONS? ROUND ----------------------------------------------
# WEIGHTS COMPARISON
path_tr_ddqn_ha3 = 'out/DDQN-HA/trajectory_set_0314-120923.pkl'
path_p_ddqn_ha3 = 'out/DDQN-HA/pax_set_0314-120923.pkl'
path_tr_ddqn_ha5 = 'out/DDQN-HA/trajectory_set_0314-120317.pkl'
path_p_ddqn_ha5 = 'out/DDQN-HA/pax_set_0314-120317.pkl'
path_tr_ddqn_ha7 = 'out/DDQN-HA/trajectory_set_0314-101720.pkl'
path_p_ddqn_ha7 = 'out/DDQN-HA/pax_set_0314-101720.pkl'
path_tr_ddqn_ha9 = 'out/DDQN-HA/trajectory_set_0303-120150.pkl' # secretly 10
path_p_ddqn_ha9 = 'out/DDQN-HA/pax_set_0303-120150.pkl' # secretly 10
path_tr_ddqn_ha11 = 'out/DDQN-HA/trajectory_set_0313-105947.pkl' # secretly 12
path_p_ddqn_ha11 = 'out/DDQN-HA/pax_set_0313-105947.pkl' # secretly 12


path_dir_w = 'out/compare/weights/'
tags_w = ['3', '5', '7', '9', '11']

# BENCHMARK COMPARISON
path_tr_nc_b = 'out/NC/trajectories_set_0302-230246.pkl'
path_p_nc_b = 'out/NC/pax_set_0302-230246.pkl'
path_tr_eh_b = 'out/EH/trajectories_set_0309-181023.pkl'
path_p_eh_b = 'out/EH/pax_set_0309-181023.pkl'
path_tr_ddqn_la_b = 'out/DDQN-LA/trajectory_set_0302-225508.pkl'
path_p_ddqn_la_b = 'out/DDQN-LA/pax_set_0302-225508.pkl'
path_tr_ddqn_ha_b = 'out/DDQN-HA/trajectory_set_0303-120150.pkl'
path_p_ddqn_ha_b = 'out/DDQN-HA/pax_set_0303-120150.pkl'
path_dir_b = 'out/compare/benchmark/'
tags_b = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA']

# FOR TRIP TIME DISTRIBUTION
path_tr_nc_t = 'out/NC/trajectories_set_0314-131714.pkl'
path_p_nc_t = 'out/NC/pax_set_0314-131714.pkl'
path_tr_eh_t = 'out/EH/trajectories_set_0314-132111.pkl'
path_p_eh_t = 'out/EH/trajectories_set_0314-132111.pkl'
path_tr_ddqn_la_t = 'out/DDQN-LA/trajectory_set_0314-130246.pkl'
path_p_ddqn_la_t = 'out/DDQN-LA/pax_set_0314-130246.pkl'
path_tr_ddqn_ha_t = 'out/DDQN-HA/trajectory_set_0314-131407.pkl'
path_p_ddqn_ha_t = 'out/DDQN-HA/pax_set_0314-131407.pkl'

# SENSITIVITY RUN TIMES
path_tr_ddqn_la_base_s1 = 'out/DDQN-LA/trajectory_set_0302-225508.pkl'
path_p_ddqn_la_base_s1 = 'out/DDQN-LA/pax_set_0302-225508.pkl'
path_tr_ddqn_la_high_s1 = 'out/DDQN-LA/trajectory_set_0306-192205.pkl'
path_p_ddqn_la_high_s1 = 'out/DDQN-LA/pax_set_0306-192205.pkl'
path_tr_ddqn_la_low_s1 = 'out/DDQN-LA/trajectory_set_0306-192238.pkl'
path_p_ddqn_la_low_s1 = 'out/DDQN-LA/pax_set_0306-192238.pkl'
path_tr_ddqn_ha_base_s1 = 'out/DDQN-HA/trajectory_set_0303-120150.pkl'
path_p_ddqn_ha_base_s1 = 'out/DDQN-HA/pax_set_0303-120150.pkl'
path_tr_ddqn_ha_high_s1 = 'out/DDQN-HA/trajectory_set_0303-134807.pkl'
path_p_ddqn_ha_high_s1 = 'out/DDQN-HA/pax_set_0303-134807.pkl'
path_tr_ddqn_ha_low_s1 = 'out/DDQN-HA/trajectory_set_0303-134728.pkl'
path_p_ddqn_ha_low_s1 = 'out/DDQN-HA/pax_set_0303-134728.pkl'
tags_s1 = ['DDQN-LA (low)', 'DDQN-HA (low)', 'DDQN-LA (medium)', 'DDQN-HA (medium)', 'DDQN-LA (high)', 'DDQN-HA (high)']
path_dir_s1 = 'out/compare/sensitivity run times/'

# SENSITIVITY COMPLIANCE
path_tr_eh_base_s2 = 'out/EH/trajectories_set_0309-181023.pkl'
path_p_eh_base_s2 = 'out/EH/pax_set_0309-181023.pkl'
path_tr_eh_80_s2 = 'out/EH/trajectories_set_0316-221132.pkl'
path_p_eh_80_s2 = 'out/EH/pax_set_0316-221132.pkl'
path_tr_eh_60_s2 = 'out/EH/trajectories_set_0316-235851.pkl'
path_p_eh_60_s2 = 'out/EH/pax_set_0316-235851.pkl'
path_tr_ddqn_la_base_s2 = 'out/DDQN-LA/trajectory_set_0302-225508.pkl'
path_p_ddqn_la_base_s2 = 'out/DDQN-LA/pax_set_0302-225508.pkl'
path_tr_ddqn_la_80_s2 = 'out/DDQN-LA/trajectory_set_0316-235624.pkl'
path_p_ddqn_la_80_s2 = 'out/DDQN-LA/pax_set_0316-235624.pkl'
path_tr_ddqn_la_60_s2 = 'out/DDQN-LA/trajectory_set_0316-223504.pkl'
path_p_ddqn_la_60_s2 = 'out/DDQN-LA/pax_set_0316-223504.pkl'
path_tr_ddqn_ha_base_s2 = 'out/DDQN-HA/trajectory_set_0303-120150.pkl' # 0222-2247
path_p_ddqn_ha_base_s2 = 'out/DDQN-HA/pax_set_0303-120150.pkl'
path_tr_ddqn_ha_80_s2 = 'out/DDQN-HA/trajectory_set_0316-213802.pkl' # 0225-1852
path_p_ddqn_ha_80_s2 = 'out/DDQN-HA/pax_set_0316-213802.pkl' # 0225-1852
path_tr_ddqn_ha_60_s2 = 'out/DDQN-HA/trajectory_set_0316-213714.pkl' # 0225-1934
path_p_ddqn_ha_60_s2 = 'out/DDQN-HA/pax_set_0316-213714.pkl' # 0225-1934
tags_s2 = ['EH (base)', 'DDQN-LA (base)', 'DDQN-HA (base)',
           'EH (0.8)', 'DDQN-LA (0.8)', 'DDQN-HA (0.8)',
           'EH (0.6)', 'DDQN-LA (0.6)', 'DDQN-HA (0.6)']
path_dir_s2 = 'out/compare/sensitivity compliance/'
