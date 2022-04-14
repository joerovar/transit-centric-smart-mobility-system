import time

path_to_ins = 'in/'
path_to_outs = 'out/'
ext_var = '.pkl'
ext_fig = '.png'
ext_csv = '.csv'
tstamp_save = time.strftime("%m%d-%H%M")
tstamp_load = tstamp_save
dir_raw = 'raw/'
dir_vis = 'vis/'
dir_figs = 'fig/'
dir_csv = 'txt/'
dir_var = 'var/'

# INPUT FILES

path_trips_gtfs = path_to_ins + dir_raw + 'gtfs/trips.txt'
path_stop_times = path_to_ins + dir_raw + 'rt20_stop_times.csv'
# path_extra_stop_times = path_to_ins + dir_raw + 'rt20_extra.csv'
path_avl = path_to_ins + dir_raw + 'rt20_avl.csv'
path_stops_loc = path_to_ins + dir_raw + 'gtfs/stops.txt'
path_od = path_to_ins + dir_raw + 'odt_for_opt.csv'
path_dispatching_times = path_to_ins + dir_raw + 'dispatching_time.csv'
path_ordered_dispatching = path_to_ins + dir_raw + 'inbound_ordered_dispatching.csv'

# EXTRACT NETWORK PARAMS

path_route_stops = path_to_ins + 'xtr/route_stops' + ext_var
path_link_times_mean = path_to_ins + 'xtr/link_times_info' + ext_var
path_arr_rates = path_to_ins + 'xtr/arr_rates' + ext_var
path_alight_fractions = path_to_ins + 'xtr/alight_fractions' + ext_var
path_alight_rates = path_to_ins + 'xtr/alight_rates' + ext_var
path_dep_volume = path_to_ins + 'xtr/dep_vol' + ext_var
path_odt_xtr = path_to_ins + 'xtr/odt' + ext_var

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
path_tr_ddqn_ha3 = 'out/DDQN-HA/0329-175147-trajectory_set.pkl'
path_p_ddqn_ha3 = 'out/DDQN-HA/0329-175147-pax_set.pkl'
path_tr_ddqn_ha5 = 'out/DDQN-HA/0329-163310-trajectory_set.pkl'
path_p_ddqn_ha5 = 'out/DDQN-HA/0329-163310-pax_set.pkl'
path_tr_ddqn_ha7 = 'out/DDQN-HA/0329-164425-trajectory_set.pkl'
path_p_ddqn_ha7 = 'out/DDQN-HA/0329-164425-pax_set.pkl'
path_tr_ddqn_ha9 = 'out/DDQN-HA/0405-014450-trajectory_set.pkl' # 0405-0120
path_p_ddqn_ha9 = 'out/DDQN-HA/0405-014450-pax_set.pkl'
path_tr_ddqn_ha11 = 'out/DDQN-HA/0329-210029-trajectory_set.pkl'
path_p_ddqn_ha11 = 'out/DDQN-HA/0329-210029-pax_set.pkl'
path_dir_w = 'out/compare/weights/'
tags_w = ['3', '5', '7', '9', '11']

# BENCHMARK COMPARISON

path_tr_nc_b = 'out/NC/0329-155354-trajectory_set.pkl'
path_p_nc_b = 'out/NC/0329-155354-pax_set.pkl'

path_tr_eh_b = 'out/EH/0329-155402-trajectory_set.pkl'
path_p_eh_b = 'out/EH/0329-155402-pax_set.pkl'

path_tr_ddqn_la_b = 'out/DDQN-LA/0329-185304-trajectory_set.pkl'
path_p_ddqn_la_b = 'out/DDQN-LA/0329-185304-pax_set.pkl'

path_tr_ddqn_ha_b = 'out/DDQN-HA/0405-014450-trajectory_set.pkl'
path_p_ddqn_ha_b = 'out/DDQN-HA/0405-014450-pax_set.pkl'

path_dir_b = 'out/compare/benchmark/'
tags_b = ['NC', 'EH', 'DDQN-LA', 'DDQN-HA']

# SENSITIVITY RUN TIMES
path_tr_eh_base_s1 = 'out/EH/0329-155402-trajectory_set.pkl'
path_p_eh_base_s1 = 'out/EH/0329-155402-pax_set.pkl'
path_tr_eh_high_s1 = 'out/EH/0329-192044-trajectory_set.pkl'
path_p_eh_high_s1 = 'out/EH/0329-192044-pax_set.pkl'
path_tr_eh_low_s1 = 'out/EH/0329-192035-trajectory_set.pkl'
path_p_eh_low_s1 = 'out/EH/0329-192035-pax_set.pkl'

path_tr_ddqn_la_base_s1 = 'out/DDQN-LA/0329-185304-trajectory_set.pkl'
path_p_ddqn_la_base_s1 = 'out/DDQN-LA/0329-185304-pax_set.pkl'
path_tr_ddqn_la_high_s1_nr = 'out/DDQN-LA/0329-190410-trajectory_set.pkl'
path_p_ddqn_la_high_s1_nr = 'out/DDQN-LA/0329-190410-pax_set.pkl'
path_tr_ddqn_la_high_s1 = 'out/DDQN-LA/0406-101143-trajectory_set.pkl'
path_p_ddqn_la_high_s1 = 'out/DDQN-LA/0406-101143-pax_set.pkl'
# path_tr_ddqn_la_low_s1_nr = 'out/DDQN-LA/0329-190330-trajectory_set.pkl'
# path_p_ddqn_la_low_s1_nr = 'out/DDQN-LA/0329-190330-pax_set.pkl'
path_tr_ddqn_la_low_s1_nr = 'out/DDQN-LA/0406-082942-trajectory_set.pkl'
path_p_ddqn_la_low_s1_nr = 'out/DDQN-LA/0406-082942-pax_set.pkl'
path_tr_ddqn_la_low_s1 = 'out/DDQN-LA/0406-124359-trajectory_set.pkl'
path_p_ddqn_la_low_s1 = 'out/DDQN-LA/0406-124359-pax_set.pkl'

# path_tr_ddqn_ha_base_s1 = 'out/DDQN-HA/0329-165457-trajectory_set.pkl'
# path_p_ddqn_ha_base_s1 = 'out/DDQN-HA/0329-165457-pax_set.pkl'
path_tr_ddqn_ha_base_s1 = 'out/DDQN-HA/0405-014450-trajectory_set.pkl'
path_p_ddqn_ha_base_s1 = 'out/DDQN-HA/0405-014450-pax_set.pkl'
path_tr_ddqn_ha_high_s1_nr = 'out/DDQN-HA/0405-114732-trajectory_set.pkl'
path_p_ddqn_ha_high_s1_nr = 'out/DDQN-HA/0405-114732-pax_set.pkl'
path_tr_ddqn_ha_high_s1 = 'out/DDQN-HA/0405-123541-trajectory_set.pkl'
path_p_ddqn_ha_high_s1 = 'out/DDQN-HA/0405-123541-pax_set.pkl'
path_tr_ddqn_ha_low_s1_nr = 'out/DDQN-HA/0405-114754-trajectory_set.pkl'
path_p_ddqn_ha_low_s1_nr = 'out/DDQN-HA/0405-114754-pax_set.pkl'
path_tr_ddqn_ha_low_s1 = 'out/DDQN-HA/0405-123518-trajectory_set.pkl'
path_p_ddqn_ha_low_s1 = 'out/DDQN-HA/0405-123518-pax_set.pkl'


tags_s1 = ['EH (low)', 'DDQN-LA (NR+low)', 'DDQN-LA (R+low)','DDQN-HA (NR+low)', 'DDQN-HA (R+low)',
           'EH (medium)', 'DDQN-LA (medium)', 'DDQN-HA (medium)',
           'EH (high)', 'DDQN-LA (NR+high)','DDQN-LA (R+high)', 'DDQN-HA (NR+high)', 'DDQN-HA (R+high)']
path_dir_s1 = 'out/compare/sensitivity run times/'

# SENSITIVITY COMPLIANCE
path_tr_eh_base_s2 = 'out/EH/0329-155402-trajectory_set.pkl'
path_p_eh_base_s2 = 'out/EH/0329-155402-pax_set.pkl'
path_tr_eh_80_s2 = 'out/EH/0324-075642-trajectory_set.pkl'
path_p_eh_80_s2 = 'out/EH/0324-075642-pax_set.pkl'
path_tr_eh_60_s2 = 'out/EH/0324-075651-trajectory_set.pkl'
path_p_eh_60_s2 = 'out/EH/0324-075651-pax_set.pkl'

path_tr_ddqn_la_base_s2 = 'out/DDQN-LA/0329-185304-trajectory_set.pkl'
path_p_ddqn_la_base_s2 = 'out/DDQN-LA/0329-185304-pax_set.pkl'
path_tr_ddqn_la_80_s2_nr = 'out/DDQN-LA/0329-193923-trajectory_set.pkl'
path_p_ddqn_la_80_s2_nr = 'out/DDQN-LA/0329-193923-pax_set.pkl'
path_tr_ddqn_la_80_s2 = 'out/DDQN-LA/0329-222510-trajectory_set.pkl'
path_p_ddqn_la_80_s2 = 'out/DDQN-LA/0329-222510-pax_set.pkl'
path_tr_ddqn_la_60_s2_nr = 'out/DDQN-LA/0329-193937-trajectory_set.pkl'
path_p_ddqn_la_60_s2_nr = 'out/DDQN-LA/0329-193937-pax_set.pkl'
path_tr_ddqn_la_60_s2 = 'out/DDQN-LA/0329-195857-trajectory_set.pkl'
path_p_ddqn_la_60_s2 = 'out/DDQN-LA/0329-195857-pax_set.pkl'


path_tr_ddqn_ha_base_s2 = 'out/DDQN-HA/0405-014450-trajectory_set.pkl'
path_p_ddqn_ha_base_s2 = 'out/DDQN-HA/0405-014450-pax_set.pkl'
path_tr_ddqn_ha_80_s2_nr = 'out/DDQN-HA/0405-102723-trajectory_set.pkl'
path_p_ddqn_ha_80_s2_nr = 'out/DDQN-HA/0405-102723-pax_set.pkl'
path_tr_ddqn_ha_80_s2 = 'out/DDQN-HA/0329-213742-trajectory_set.pkl'
path_p_ddqn_ha_80_s2 = 'out/DDQN-HA/0329-213742-pax_set.pkl'
path_tr_ddqn_ha_60_s2_nr = 'out/DDQN-HA/0405-102850-trajectory_set.pkl'
path_p_ddqn_ha_60_s2_nr = 'out/DDQN-HA/0405-102850-pax_set.pkl'
path_tr_ddqn_ha_60_s2 = 'out/DDQN-HA/0329-204648-trajectory_set.pkl'
path_p_ddqn_ha_60_s2 = 'out/DDQN-HA/0329-204648-pax_set.pkl'
tags_s2 = ['EH (base)', 'DDQN-LA (base)', 'DDQN-HA (base)',
           'EH (0.8)', 'DDQN-LA (NR+0.8)', 'DDQN-LA (0.8)', 'DDQN-HA (NR+0.8)', 'DDQN-HA (0.8)',
           'EH (0.6)', 'DDQN-LA (NR+0.6)','DDQN-LA (0.6)', 'DDQN-HA (NR+0.6)','DDQN-HA (0.6)']
path_dir_s2 = 'out/compare/sensitivity compliance/'
