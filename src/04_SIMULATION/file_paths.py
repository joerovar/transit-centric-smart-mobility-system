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
path_extra_stop_times = path_to_ins + dir_raw + 'rt20_extra.csv'
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

# WEIGHTS COMPARISON
path_tr_ddqn_ha3 = 'out/DDQN-HA/0321-212306-trajectory_set.pkl'
path_p_ddqn_ha3 = 'out/DDQN-HA/0321-212306-pax_set.pkl'
path_tr_ddqn_ha5 = 'out/DDQN-HA/0322-070258-trajectory_set.pkl'
path_p_ddqn_ha5 = 'out/DDQN-HA/0322-070258-pax_set.pkl'
path_tr_ddqn_ha7 = 'out/DDQN-HA/0322-220823-trajectory_set.pkl'
path_p_ddqn_ha7 = 'out/DDQN-HA/0322-220823-pax_set.pkl'
path_tr_ddqn_ha9 = 'out/DDQN-HA/0323-151033-trajectory_set.pkl'
path_p_ddqn_ha9 = 'out/DDQN-HA/0323-151033-pax_set.pkl'
path_tr_ddqn_ha11 = 'out/DDQN-HA/0322-171049-trajectory_set.pkl'
path_p_ddqn_ha11 = 'out/DDQN-HA/0322-171049-pax_set.pkl'

path_dir_w = 'out/compare/weights/'
tags_w = ['3', '5', '7', '9', '11']

# BENCHMARK COMPARISON

# path_tr_nc_b = 'out/NC/0321-202913-trajectory_set.pkl'
# path_p_nc_b = 'out/NC/0321-202913-pax_set.pkl'
#
# path_tr_eh_b = 'out/EH/0321-203010-trajectory_set.pkl'
# path_p_eh_b = 'out/EH/0321-203010-pax_set.pkl'

path_tr_nc_b = 'out/NC/0328-213908-trajectory_set.pkl'
path_p_nc_b = 'out/NC/0328-213908-pax_set.pkl'

path_tr_eh_b = 'out/EH/0328-213920-trajectory_set.pkl'
path_p_eh_b = 'out/EH/0328-213920-pax_set.pkl'

path_tr_ddqn_la_b = 'out/DDQN-LA/0323-160821-trajectory_set.pkl'
path_p_ddqn_la_b = 'out/DDQN-LA/0323-160821-pax_set.pkl'

path_tr_ddqn_ha_b = 'out/DDQN-HA/0323-151033-trajectory_set.pkl'
path_p_ddqn_ha_b = 'out/DDQN-HA/0323-151033-pax_set.pkl'

path_dir_b = 'out/compare/benchmark/'
tags_b = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA']

# SENSITIVITY RUN TIMES
path_tr_ddqn_la_base_s1 = 'out/DDQN-LA/0323-160821-trajectory_set.pkl'
path_p_ddqn_la_base_s1 = 'out/DDQN-LA/0323-160821-pax_set.pkl'
path_tr_ddqn_la_high_s1 = 'out/DDQN-LA/0323-180055-trajectory_set.pkl'
path_p_ddqn_la_high_s1 = 'out/DDQN-LA/0323-180055-pax_set.pkl'
path_tr_ddqn_la_low_s1 = 'out/DDQN-LA/0323-163123-trajectory_set.pkl'
path_p_ddqn_la_low_s1 = 'out/DDQN-LA/0323-163123-pax_set.pkl'

path_tr_ddqn_ha_base_s1 = 'out/DDQN-HA/0323-151033-trajectory_set.pkl'
path_p_ddqn_ha_base_s1 = 'out/DDQN-HA/0323-151033-pax_set.pkl'
path_tr_ddqn_ha_high_s1 = 'out/DDQN-HA/0328-154752-trajectory_set.pkl'
path_p_ddqn_ha_high_s1 = 'out/DDQN-HA/0328-154752-pax_set.pkl'
path_tr_ddqn_ha_low_s1 = 'out/DDQN-HA/0323-163426-trajectory_set.pkl'
path_p_ddqn_ha_low_s1 = 'out/DDQN-HA/0323-163426-pax_set.pkl'
tags_s1 = ['DDQN-LA (low)', 'DDQN-HA (low)', 'DDQN-LA (medium)', 'DDQN-HA (medium)', 'DDQN-LA (high)', 'DDQN-HA (high)']
path_dir_s1 = 'out/compare/sensitivity run times/'

# SENSITIVITY COMPLIANCE
path_tr_eh_base_s2 = 'out/EH/0321-203010-trajectory_set.pkl'
path_p_eh_base_s2 = 'out/EH/0321-203010-pax_set.pkl'
path_tr_eh_80_s2 = 'out/EH/0324-075642-trajectory_set.pkl'
path_p_eh_80_s2 = 'out/EH/0324-075642-pax_set.pkl'
path_tr_eh_60_s2 = 'out/EH/0324-075651-trajectory_set.pkl'
path_p_eh_60_s2 = 'out/EH/0324-075651-pax_set.pkl'

path_tr_ddqn_la_base_s2 = 'out/DDQN-LA/0323-160821-trajectory_set.pkl'
path_p_ddqn_la_base_s2 = 'out/DDQN-LA/0323-160821-pax_set.pkl'
path_tr_ddqn_la_80_s2_nr = 'out/DDQN-LA/0328-202345-trajectory_set.pkl'
path_p_ddqn_la_80_s2_nr = 'out/DDQN-LA/0328-202345-pax_set.pkl'
path_tr_ddqn_la_80_s2 = 'out/DDQN-LA/0324-080452-trajectory_set.pkl'
path_p_ddqn_la_80_s2 = 'out/DDQN-LA/0324-080452-pax_set.pkl'
path_tr_ddqn_la_60_s2_nr = 'out/DDQN-LA/0328-202415-trajectory_set.pkl'
path_p_ddqn_la_60_s2_nr = 'out/DDQN-LA/0328-202415-pax_set.pkl'
path_tr_ddqn_la_60_s2 = 'out/DDQN-LA/0328-202042-trajectory_set.pkl'
path_p_ddqn_la_60_s2 = 'out/DDQN-LA/0328-202042-pax_set.pkl'

path_tr_ddqn_ha_base_s2 = 'out/DDQN-HA/0323-151033-trajectory_set.pkl'
path_p_ddqn_ha_base_s2 = 'out/DDQN-HA/0323-151033-pax_set.pkl'
path_tr_ddqn_ha_80_s2_nr = 'out/DDQN-HA/0328-202826-trajectory_set.pkl'
path_p_ddqn_ha_80_s2_nr = 'out/DDQN-HA/0328-202826-pax_set.pkl'
# path_tr_ddqn_ha_80_s2 = 'out/DDQN-HA/0323-231410-trajectory_set.pkl'
# path_p_ddqn_ha_80_s2 = 'out/DDQN-HA/0323-231410-pax_set.pkl'
path_tr_ddqn_ha_80_s2 = 'out/DDQN-HA/0328-210729-trajectory_set.pkl'
path_p_ddqn_ha_80_s2 = 'out/DDQN-HA/0328-210729-pax_set.pkl'
path_tr_ddqn_ha_60_s2_nr = 'out/DDQN-HA/0328-202851-trajectory_set.pkl'
path_p_ddqn_ha_60_s2_nr = 'out/DDQN-HA/0328-202851-pax_set.pkl'
path_tr_ddqn_ha_60_s2 = 'out/DDQN-HA/0324-091340-trajectory_set.pkl'
path_p_ddqn_ha_60_s2 = 'out/DDQN-HA/0324-091340-pax_set.pkl'
tags_s2 = ['EH (base)', 'DDQN-LA (base)', 'DDQN-HA (base)',
           'EH (0.8)', 'DDQN-LA (NR,0.8)', 'DDQN-LA (0.8)', 'DDQN-HA (NR, 0.8)', 'DDQN-HA (0.8)',
           'EH (0.6)', 'DDQN-LA (NR, 0.6)','DDQN-LA (0.6)', 'DDQN-HA (NR, 0.6)','DDQN-HA (0.6)']
path_dir_s2 = 'out/compare/sensitivity compliance/'
