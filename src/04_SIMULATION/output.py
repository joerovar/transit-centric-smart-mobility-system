import post_process
from input import *
from os import listdir
from os.path import isfile, join

trajectories_filename = 'trajectories_'
wait_time_filename = 'wait_time_'
link_time_filename = 'link_time_'
headway_filename = 'headway_'
load_profile_filename = 'load_profile_'
denied_filename = 'denied_'
hold_time_filename = 'hold_time_'
hold_time_distribution_filename = 'hold_time_distribution_'
sars_record_filename = 'sars_record_'
dwell_time_filename = 'dwell_time_'
od_jt_mean_filename = 'od_jt_m_'
od_wt_mean_filename = 'od_wt_m_'
od_jt_std_filename = 'od_jt_s_'
od_wt_std_filename = 'od_wt_s_'
travel_time_dist_filename = 'travel_time_dist_'

# parameters for benchmark comparison
# cv headway
hw_nc_varname = 'cv_hw_nc'
hw_eh_varname = 'cv_hw_eh'
hw_rl_varname = 'cv_hw_rl'
# travel time dist
ttd_nc_varname = 'ttd_nc'
ttd_eh_varname = 'ttd_eh'
ttd_rl_varname = 'ttd_rl'
# od wait time mean
od_wtm_nc_varname = 'od_wtm_nc'
od_wtm_eh_varname = 'od_wtm_eh'
od_wtm_rl_varname = 'od_wtm_rl'
# od wait time standard deviation
od_wts_nc_varname = 'od_wts_nc'
od_wts_eh_varname = 'od_wts_eh'
od_wts_rl_varname = 'od_wts_rl'
# od journey time mean
od_jtm_nc_varname = 'od_jtm_nc'
od_jtm_eh_varname = 'od_jtm_eh'
od_jtm_rl_varname = 'od_jtm_rl'
# od journey time std
od_jts_nc_varname = 'od_jts_nc'
od_jts_eh_varname = 'od_jts_eh'
od_jts_rl_varname = 'od_jts_rl'


def get_results(tstamps):
    # FULLY TRANSITIONED TO OD FLOW MODELING.
    trajectories_set = []
    completed_pax_set = []
    first_trip_id = ORDERED_TRIPS[0]
    for t in tstamps:
        # --------------------------------------------- TRAJECTORIES -----------------------
        path_trajectories_load = path_to_outs + dir_var + trajectories_filename + t + ext_var
        trajectories = post_process.load(path_trajectories_load)

        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        trajectories_header = ['trip_id', 'stop_id', 'arr_t', 'dep_t', 'load_count',
                               'ons_count', 'offs_count', 'denied_count']
        path_trajectories_write = path_to_outs + dir_csv + trajectories_filename + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + trajectories_filename + t + ext_fig
        # post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        # post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS)

        # ------------------------------------------------- PAX DATA ---------------------------
        path_completed_pax_load = path_to_outs + dir_var + 'completed_pax_' + t + ext_var
        completed_pax_set.append(post_process.load(path_completed_pax_load))

    if len(tstamps) > 1:
        t = tstamps[-1]

        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED,
                                                                                              first_trip_id)
        stops_loc = post_process.get_stop_loc(path_stops_loc)

        # ------------------------------------------------ WAIT TIMES -----------------------------------------
        path_wtimes_write = path_to_outs + dir_csv + wait_time_filename + t + ext_csv
        path_plot_wtime_combined = path_to_outs + dir_figs + wait_time_filename + t + ext_fig

        t_per_stop_lbl = ['stop id', 'seconds']

        post_process.write_wait_times(wait_time_comb, stops_loc, path_wtimes_write, STOPS)
        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl)

        # ------------------------------------------------ LINK TIMES/DWELL TIMES -------------------------------------
        path_ltimes_write = path_to_outs + dir_csv + link_time_filename + t + ext_csv
        path_ltimes_plot = path_to_outs + dir_figs + link_time_filename + t + ext_fig
        path_dtimes_write = path_to_outs + dir_csv + dwell_time_filename + t + ext_csv
        path_dtimes_plot = path_to_outs + dir_figs + dwell_time_filename + t + ext_fig

        ltimes_mean, ltimes_std, dtimes_mean, \
        dtimes_std, segment_times = post_process.travel_times_from_trajectory_set(trajectories_set, IDX_DEP_T,
                                                                                  IDX_ARR_T, first_trip_id, TPOINT0,
                                                                                  TPOINT1)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, STOPS, path_dtimes_plot, ltimes_lbl,
                                      x_y_lbls=ltimes_x_y_lbls)

        # ------------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig
        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS)

        # ------------------------------------------------ LOAD PROFILE ----------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP, first_trip_id)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        load_labels = ['stop id', 'pax per trip', 'pax load']

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined, x_y_lbls=load_labels)

        # ------------------------------------------------- DENIED BOARDINGS -----------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb,
                                                                        first_trip_id)
        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        denied_labels = ['stop id', '1 in 1,000 pax']

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=denied_labels)

        # ------------------------------------------------ OD-LEVEL DATA --------------------------------------
        od_journey_time_mean, od_journey_time_std, od_wait_time_mean, \
        od_wait_time_std = post_process.process_od_level_data(completed_pax_set, STOPS, FOCUS_START_TIME_SEC)
        path_plot_od_jt_mean = path_to_outs + dir_figs + od_jt_mean_filename + t + ext_fig
        path_plot_od_jt_std = path_to_outs + dir_figs + od_jt_std_filename + t + ext_fig
        path_plot_od_wt_mean = path_to_outs + dir_figs + od_wt_mean_filename + t + ext_fig
        path_plot_od_wt_std = path_to_outs + dir_figs + od_wt_std_filename + t + ext_fig

        # post_process.plot_od(od_journey_time_mean, path_plot_od_jt_mean)
        # post_process.plot_od(od_journey_time_std, path_plot_od_jt_std)
        # post_process.plot_od(od_wait_time_mean, path_plot_od_wt_mean)
        # post_process.plot_od(od_wait_time_std, path_plot_od_wt_std)

        # ----------------------------------------- SAVE DATA FOR BENCHMARK COMPARISON ------------------------------
        post_process.save(path_to_outs + dir_var + hw_nc_varname + ext_var, headway_comb)
        post_process.save(path_to_outs + dir_var + ttd_nc_varname + ext_var, segment_times)
        post_process.save(path_to_outs + dir_var + od_wtm_nc_varname + ext_var, od_wait_time_mean)
        post_process.save(path_to_outs + dir_var + od_wts_nc_varname + ext_var, od_wait_time_std)
        post_process.save(path_to_outs + dir_var + od_jtm_nc_varname + ext_var, od_journey_time_mean)
        post_process.save(path_to_outs + dir_var + od_jts_nc_varname + ext_var, od_journey_time_std)
    return


def get_base_control_results(tstamps):
    # THINGS TO COMBINE
    trajectories_set = []
    completed_pax_set = []
    first_trip_id = ORDERED_TRIPS[0]
    for t in tstamps:
        # -------------------------------------------- TRAJECTORIES ---------------------------------------------
        path_trajectories_load = path_to_outs + dir_var + trajectories_filename + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        trajectories_header = ['trip_id', 'stop_id', 'arr_t', 'dep_t', 'load_count', 'ons_count',
                               'offs_count', 'denied_count', 'hold_sec']
        path_trajectories_write = path_to_outs + dir_csv + trajectories_filename + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + trajectories_filename + t + ext_fig
        # post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        # post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS,
        #                                controlled_stops=CONTROLLED_STOPS)
        # ------------------------------------------------- PAX DATA ---------------------------
        path_completed_pax_load = path_to_outs + dir_var + 'completed_pax_' + t + ext_var
        completed_pax_set.append(post_process.load(path_completed_pax_load))

    if len(tstamps) > 1:
        t = tstamps[-1]

        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED,
                                                                                              first_trip_id)
        stops_loc = post_process.get_stop_loc(path_stops_loc)

        # ------------------------------------------ WAIT TIMES ---------------------------------------------------

        path_wtimes_write = path_to_outs + dir_csv + wait_time_filename + t + ext_csv
        path_plot_wtime_combined = path_to_outs + dir_figs + wait_time_filename + t + ext_fig

        post_process.write_wait_times(wait_time_comb, stops_loc, path_wtimes_write, STOPS)

        t_per_stop_lbl = ['stop id', 'seconds']

        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl,
                                    controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LINK TIMES ----------------------------------------------------
        path_ltimes_write = path_to_outs + dir_csv + link_time_filename + t + ext_csv
        path_ltimes_plot = path_to_outs + dir_figs + link_time_filename + t + ext_fig
        path_dtimes_write = path_to_outs + dir_csv + dwell_time_filename + t + ext_csv
        path_dtimes_plot = path_to_outs + dir_figs + dwell_time_filename + t + ext_fig

        ltimes_mean, ltimes_std, dtimes_mean, \
        dtimes_std, segment_times = post_process.travel_times_from_trajectory_set(trajectories_set, IDX_DEP_T,
                                                                                  IDX_ARR_T, first_trip_id, TPOINT0,
                                                                                  TPOINT1)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, STOPS, path_dtimes_plot, ltimes_lbl,
                                      x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig

        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS,
                                  controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LOAD PROFILE ------------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP, first_trip_id)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'pax per trip', 'pax load'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- DENIED BOARDINGS -----------------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb,
                                                                        first_trip_id)

        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', '1 in 1,000 pax'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- HOLD TIME ---------------------------------------------
        path_plot_holding_time_comb = path_to_outs + dir_figs + hold_time_filename + t + ext_fig
        path_plot_holding_time_distribution = path_to_outs + dir_figs + hold_time_distribution_filename + t + ext_fig
        hold_time_comb, hold_time_all = post_process.hold_time_from_trajectory_set(trajectories_set, IDX_HOLD_TIME,
                                                                                   first_trip_id, CONTROLLED_STOPS[:-1])

        post_process.plot_bar_chart(hold_time_comb, STOPS, path_plot_holding_time_comb,
                                    x_y_lbls=['stop id', 'seconds'], controlled_stops=CONTROLLED_STOPS)

        post_process.plot_histogram(hold_time_all, path_plot_holding_time_distribution)

        # ------------------------------------------------ OD-LEVEL DATA --------------------------------------
        od_journey_time_mean, od_journey_time_std, od_wait_time_mean, \
        od_wait_time_std = post_process.process_od_level_data(completed_pax_set, STOPS, FOCUS_START_TIME_SEC)
        # path_plot_od_jt_mean = path_to_outs + dir_figs + od_jt_mean_filename + t + ext_fig
        # path_plot_od_jt_std = path_to_outs + dir_figs + od_jt_std_filename + t + ext_fig
        # path_plot_od_wt_mean = path_to_outs + dir_figs + od_wt_mean_filename + t + ext_fig
        # path_plot_od_wt_std = path_to_outs + dir_figs + od_wt_std_filename + t + ext_fig

        # post_process.plot_od(od_journey_time_mean, path_plot_od_jt_mean)
        # post_process.plot_od(od_journey_time_std, path_plot_od_jt_std)
        # post_process.plot_od(od_wait_time_mean, path_plot_od_wt_mean)
        # post_process.plot_od(od_wait_time_std, path_plot_od_wt_std)

        # ----------------------------------------- SAVE DATA FOR BENCHMARK COMPARISON ------------------------------
        post_process.save(path_to_outs + dir_var + hw_eh_varname + ext_var, headway_comb)
        post_process.save(path_to_outs + dir_var + ttd_eh_varname + ext_var, segment_times)
        post_process.save(path_to_outs + dir_var + od_wtm_eh_varname + ext_var, od_wait_time_mean)
        post_process.save(path_to_outs + dir_var + od_wts_eh_varname + ext_var, od_wait_time_std)
        post_process.save(path_to_outs + dir_var + od_jtm_eh_varname + ext_var, od_journey_time_mean)
        post_process.save(path_to_outs + dir_var + od_jts_eh_varname + ext_var, od_journey_time_std)
    return


def get_rl_results(tstamps):
    # THINGS TO COMBINE
    trajectories_set = []
    completed_pax_set = []
    first_trip_id = ORDERED_TRIPS[0]
    for t in tstamps:
        # ---------------------------------------- SARS ------------------------------------------------------
        path_sars = path_to_outs + dir_var + sars_record_filename + t + ext_var
        sars = post_process.load(path_sars)

        path_trajectories_load = path_to_outs + dir_var + trajectories_filename + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        # ----------------------------------------- TRAJECTORIES --------------------------------------------
        trajectories_header = ['trip_id', 'stop_id', 'arr_t', 'dep_t', 'load_count', 'ons_count', 'offs_count',
                               'denied_count', 'hold_sec', 'skip']
        path_trajectories_write = path_to_outs + dir_csv + trajectories_filename + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + trajectories_filename + t + ext_fig
        path_sars_write = path_to_outs + dir_csv + sars_record_filename + t + ext_csv
        post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS,
                                       controlled_stops=CONTROLLED_STOPS)
        post_process.write_trajectories(sars, path_sars_write)

        # ------------------------------------------------- PAX DATA ---------------------------
        path_completed_pax_load = path_to_outs + dir_var + 'completed_pax_' + t + ext_var
        completed_pax_set.append(post_process.load(path_completed_pax_load))

    if len(tstamps) > 1:
        t = tstamps[-1]
        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED,
                                                                                              first_trip_id)
        stops_loc = post_process.get_stop_loc(path_stops_loc)

        # ------------------------------------------ WAIT TIMES ---------------------------------------------------

        path_wtimes_write = path_to_outs + dir_csv + wait_time_filename + t + ext_csv
        path_plot_wtime_combined = path_to_outs + dir_figs + wait_time_filename + t + ext_fig

        post_process.write_wait_times(wait_time_comb, stops_loc, path_wtimes_write, STOPS)

        t_per_stop_lbl = ['stop id', 'seconds']

        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl,
                                    controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LINK/DWELL TIMES ----------------------------------------------
        path_ltimes_write = path_to_outs + dir_csv + link_time_filename + t + ext_csv
        path_ltimes_plot = path_to_outs + dir_figs + link_time_filename + t + ext_fig
        path_dtimes_write = path_to_outs + dir_csv + dwell_time_filename + t + ext_csv
        path_dtimes_plot = path_to_outs + dir_figs + dwell_time_filename + t + ext_fig

        ltimes_mean, ltimes_std, dtimes_mean, \
        dtimes_std, segment_times = post_process.travel_times_from_trajectory_set(trajectories_set, IDX_DEP_T,
                                                                                  IDX_ARR_T, first_trip_id, TPOINT0,
                                                                                  TPOINT1)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, STOPS, path_dtimes_plot, ltimes_lbl,
                                      x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig

        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS,
                                  controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LOAD PROFILE ------------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP, first_trip_id)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'pax per trip', 'pax load'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- DENIED BOARDINGS -----------------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb,
                                                                        first_trip_id)

        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', '1 in 1,000 pax'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- HOLDING TIME ---------------------------------------------
        path_plot_holding_time_comb = path_to_outs + dir_figs + hold_time_filename + t + ext_fig
        path_plot_hold_time_distribution = path_to_outs + dir_figs + hold_time_distribution_filename + t + ext_fig
        hold_time_comb, hold_time_all = post_process.hold_time_from_trajectory_set(trajectories_set, IDX_HOLD_TIME,
                                                                                   first_trip_id, CONTROLLED_STOPS[:-1])
        post_process.plot_bar_chart(hold_time_comb, STOPS, path_plot_holding_time_comb,
                                    x_y_lbls=['stop id', 'seconds'], controlled_stops=CONTROLLED_STOPS)
        post_process.plot_histogram(hold_time_all, path_plot_hold_time_distribution)
        # ------------------------------------------------ OD-LEVEL DATA --------------------------------------
        od_journey_time_mean, od_journey_time_std, od_wait_time_mean, \
        od_wait_time_std = post_process.process_od_level_data(completed_pax_set, STOPS, FOCUS_START_TIME_SEC)
        path_plot_od_jt_mean = path_to_outs + dir_figs + od_jt_mean_filename + t + ext_fig
        path_plot_od_jt_std = path_to_outs + dir_figs + od_jt_std_filename + t + ext_fig
        path_plot_od_wt_mean = path_to_outs + dir_figs + od_wt_mean_filename + t + ext_fig
        path_plot_od_wt_std = path_to_outs + dir_figs + od_wt_std_filename + t + ext_fig

        # post_process.plot_od(od_journey_time_mean, path_plot_od_jt_mean)
        # post_process.plot_od(od_journey_time_std, path_plot_od_jt_std)
        # post_process.plot_od(od_wait_time_mean, path_plot_od_wt_mean)
        # post_process.plot_od(od_wait_time_std, path_plot_od_wt_std)

        # ----------------------------------------- SAVE DATA FOR BENCHMARK COMPARISON ------------------------------
        post_process.save(path_to_outs + dir_var + hw_rl_varname + ext_var, headway_comb)
        post_process.save(path_to_outs + dir_var + ttd_rl_varname + ext_var, segment_times)
        post_process.save(path_to_outs + dir_var + od_wtm_rl_varname + ext_var, od_wait_time_mean)
        post_process.save(path_to_outs + dir_var + od_wts_rl_varname + ext_var, od_wait_time_std)
        post_process.save(path_to_outs + dir_var + od_jtm_rl_varname + ext_var, od_journey_time_mean)
        post_process.save(path_to_outs + dir_var + od_jts_rl_varname + ext_var, od_journey_time_std)
        return mean_load_comb

    else:
        return


def access_past_results(path_dir_load, vartype, tstamp_contained, path_dir_save):
    onlyfiles = [f for f in listdir(path_dir_load) if
                 isfile(join(path_dir_load, f)) and vartype in f and tstamp_contained in f]
    for f in onlyfiles:
        path_load_file = path_dir_load + '/' + f
        var = post_process.load(path_load_file)
        path_save_file = path_dir_save + '/' + f
        path_sars_write = path_save_file.replace(ext_var, ext_csv)
        post_process.write_trajectories(var, path_sars_write)
    return


def benchmark_comparisons():
    lbls = ['NC', 'EH', 'RL']
    # headway graph

    # travel time graph

    # wait time mean RL vs NC

    # wait time std RL vs NC

    # wait time mean RL vs EH

    # wait time std RL vs EH

    return


# access_past_results('out/var', 'sars_record', '1201', 'out/txt')
