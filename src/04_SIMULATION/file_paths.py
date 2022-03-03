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

# 40 REPLICATIONS ROUND ----------------------------------
# # BENCHMARK COMPARISON
# path_tr_nc_b = 'out/NC/trajectories_set_0222-211726.pkl'
# path_p_nc_b = 'out/NC/pax_set_0222-211726.pkl'
# path_tr_eh_b = 'out/EH/trajectories_set_0222-234859.pkl'
# path_p_eh_b = 'out/EH/pax_set_0222-234859.pkl'
# path_tr_ddqn_la_b = 'out/DDQN-LA/trajectory_set_0224-124312.pkl' # 0224-1234
# path_p_ddqn_la_b = 'out/DDQN-LA/pax_set_0224-124312.pkl' # 0224-1234
# path_tr_ddqn_ha1_b = 'out/DDQN-HA/trajectory_set_0222-233419.pkl' # 0222-2247
# path_p_ddqn_ha1_b = 'out/DDQN-HA/pax_set_0222-233419.pkl'
# path_tr_ddqn_ha2_b = 'out/DDQN-HA/trajectory_set_0223-183027.pkl' # 0223-1249
# path_p_ddqn_ha2_b = 'out/DDQN-HA/pax_set_0223-183027.pkl'
# path_tr_ddqn_ha3_b = 'out/DDQN-HA/trajectory_set_0222-233609.pkl' # 0222-2315
# path_p_ddqn_ha3_b = 'out/DDQN-HA/pax_set_0222-233609.pkl'
# path_tr_ddqn_ha4_b = 'out/DDQN-HA/trajectory_set_0223-220817.pkl' # 0223-2159
# path_p_ddqn_ha4_b = 'out/DDQN-HA/pax_set_0223-220817.pkl'
# path_dir_b = 'out/compare/benchmark/'
# tags_b = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA (NO RT)', 'DDQN-HA (3.0)', 'DDQN-HA (1.5)', 'DDQN-HA (1.0)']

# SENSITIVITY RUN TIMES
path_tr_ddqn_la_base_s1 = 'out/DDQN-LA/trajectory_set_0224-124312.pkl'
path_p_ddqn_la_base_s1 = 'out/DDQN-LA/pax_set_0224-124312.pkl'
path_tr_ddqn_la_high_s1 = 'out/DDQN-LA/trajectory_set_0225-153557.pkl'
path_p_ddqn_la_high_s1 = 'out/DDQN-LA/pax_set_0225-153557.pkl'
path_tr_ddqn_la_low_s1 = 'out/DDQN-LA/trajectory_set_0225-153624.pkl'
path_p_ddqn_la_low_s1 = 'out/DDQN-LA/pax_set_0225-153624.pkl'
path_tr_ddqn_ha_base_s1 = 'out/DDQN-HA/trajectory_set_0223-183027.pkl'
path_p_ddqn_ha_base_s1 = 'out/DDQN-HA/pax_set_0223-183027.pkl'
path_tr_ddqn_ha_high_s1 = 'out/DDQN-HA/trajectory_set_0225-145011.pkl'
path_p_ddqn_ha_high_s1 = 'out/DDQN-HA/pax_set_0225-145011.pkl'
path_tr_ddqn_ha_low_s1 = 'out/DDQN-HA/trajectory_set_0225-145143.pkl'
path_p_ddqn_ha_low_s1 = 'out/DDQN-HA/pax_set_0225-145143.pkl'
tags_s1 = ['DDQN-LA (low)', 'DDQN-HA (low)', 'DDQN-LA (medium)', 'DDQN-HA (medium)', 'DDQN-LA (high)', 'DDQN-HA (high)']
path_dir_s1 = 'out/compare/sensitivity run times/'

# SENSITIVITY COMPLIANCE
path_tr_ddqn_la_base_s2 = 'out/DDQN-LA/trajectory_set_0224-124312.pkl'
path_p_ddqn_la_base_s2 = 'out/DDQN-LA/pax_set_0224-124312.pkl'
path_tr_ddqn_la_10_s2 = 'out/DDQN-LA/trajectory_set_0225-184023.pkl' # 0225-1827
path_p_ddqn_la_10_s2 = 'out/DDQN-LA/pax_set_0225-184023.pkl' # 0225-1827
path_tr_ddqn_la_20_s2 = 'out/DDQN-LA/trajectory_set_0225-181602.pkl' # 0225-1755
path_p_ddqn_la_20_s2 = 'out/DDQN-LA/pax_set_0225-181602.pkl' # 0225-1755
path_tr_ddqn_ha_base_s2 = 'out/DDQN-HA/trajectory_set_0223-183027.pkl' # 0222-2247
path_p_ddqn_ha_base_s2 = 'out/DDQN-HA/pax_set_0223-183027.pkl'
path_tr_ddqn_ha_10_s2 = 'out/DDQN-HA/trajectory_set_0225-191414.pkl' # 0225-1852
path_p_ddqn_ha_10_s2 = 'out/DDQN-HA/pax_set_0225-191414.pkl' # 0225-1852
path_tr_ddqn_ha_20_s2 = 'out/DDQN-HA/trajectory_set_0225-194326.pkl' # 0225-1934
path_p_ddqn_ha_20_s2 = 'out/DDQN-HA/pax_set_0225-194326.pkl' # 0225-1934
tags_s2 = ['DDQN-LA (0%)', 'DDQN-HA (0%)', 'DDQN-LA(10%)', 'DDQN-HA(10%)', 'DDQN-LA(20%)', 'DDQN-HA(20%)']
path_dir_s2 = 'out/compare/sensitivity compliance/'

# 70 REPLICATIONS? ROUND ----------------------------------------------
# BENCHMARK COMPARISON
path_tr_nc_b = 'out/NC/trajectories_set_0302-230246.pkl'
path_p_nc_b = 'out/NC/pax_set_0302-230246.pkl'
path_tr_eh_b = 'out/EH/trajectories_set_0302-230307.pkl'
path_p_eh_b = 'out/EH/pax_set_0302-230307.pkl'
path_tr_ddqn_la_b = 'out/DDQN-LA/trajectory_set_0302-225508.pkl' # 0224-1234
path_p_ddqn_la_b = 'out/DDQN-LA/pax_set_0302-225508.pkl' # 0224-1234
path_tr_ddqn_ha1_b = 'out/DDQN-HA/trajectory_set_0303-124521.pkl' # 0303-1023
path_p_ddqn_ha1_b = 'out/DDQN-HA/pax_set_0303-124521.pkl'
path_tr_ddqn_ha2_b = 'out/DDQN-HA/trajectory_set_0303-124446.pkl' # 0303-1014
path_p_ddqn_ha2_b = 'out/DDQN-HA/pax_set_0303-124446.pkl'
path_tr_ddqn_ha3_b = 'out/DDQN-HA/trajectory_set_0303-120044.pkl' # 0303-1059
path_p_ddqn_ha3_b = 'out/DDQN-HA/pax_set_0303-120044.pkl'
path_tr_ddqn_ha4_b = 'out/DDQN-HA/trajectory_set_0303-120150.pkl' # 0303-0943
path_p_ddqn_ha4_b = 'out/DDQN-HA/pax_set_0303-120150.pkl'
path_dir_b = 'out/compare/benchmark/'
tags_b = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA (2)', 'DDQN-HA (3)', 'DDQN-HA (5)', 'DDQN-HA (7)']