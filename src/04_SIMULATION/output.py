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


def get_results(tstamps):
    trajectories_set = []
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
        post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS)

    if len(tstamps) > 1:
        t = tstamps[-1]

        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
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

        ltimes_mean, ltimes_std, dtimes_mean, dtimes_std = post_process.travel_times_from_trajectory_set(trajectories_set, IDX_DEP_T, IDX_ARR_T)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, STOPS, path_dtimes_plot, ltimes_lbl, x_y_lbls=ltimes_x_y_lbls)

        # ------------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig
        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS)

        # ------------------------------------------------ LOAD PROFILE ----------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        load_labels = ['stop id', 'pax per trip', 'pax load']

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined, x_y_lbls=load_labels)

        # ------------------------------------------------- DENIED BOARDINGS -----------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb)
        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        denied_labels = ['stop id', '1 in 1,000 pax']

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=denied_labels)
    return


def get_base_control_results(tstamps):
    # THINGS TO COMBINE
    trajectories_set = []
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
        post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS,
                                       controlled_stops=CONTROLLED_STOPS)

    if len(tstamps) > 1:
        t = tstamps[-1]

        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
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
        path_dtimes_write = path_to_outs + dir_figs + dwell_time_filename + t + ext_csv
        path_dtimes_plot = path_to_outs + dir_figs + dwell_time_filename + t + ext_fig

        ltimes_mean, ltimes_std, dtimes_mean, dtimes_std = post_process.travel_times_from_trajectory_set(
            trajectories_set, IDX_DEP_T, IDX_ARR_T)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_plot, STOPS,
                                      controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig

        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS,
                                  controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LOAD PROFILE ------------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'pax per trip', 'pax load'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- DENIED BOARDINGS -----------------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb)

        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', '1 in 1,000 pax'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- HOLD TIME ---------------------------------------------
        path_plot_holding_time_comb = path_to_outs + dir_figs + hold_time_filename + t + ext_fig
        path_plot_holding_time_distribution = path_to_outs + dir_figs + hold_time_distribution_filename + t + ext_fig
        hold_time_comb, hold_time_all = post_process.hold_time_from_trajectory_set(trajectories_set, IDX_HOLD_TIME)

        post_process.plot_bar_chart(hold_time_comb, STOPS, path_plot_holding_time_comb,
                                    x_y_lbls=['stop id', 'seconds'], controlled_stops=CONTROLLED_STOPS)

        post_process.plot_histogram(hold_time_all, path_plot_holding_time_distribution)

    return


def get_rl_results(tstamps):
    # THINGS TO COMBINE
    trajectories_set = []
    for t in tstamps:
        # ---------------------------------------- SARS ------------------------------------------------------
        path_sars = path_to_outs + dir_var + sars_record_filename + t + ext_var
        sars = post_process.load(path_sars)

        path_trajectories_load = path_to_outs + dir_var + trajectories_filename + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        trajectories_header = ['trip_id', 'stop_id', 'arr_t', 'dep_t', 'load_count', 'ons_count', 'offs_count',
                               'denied_count', 'hold_sec', 'skip']
        path_trajectories_write = path_to_outs + dir_csv + trajectories_filename + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + trajectories_filename + t + ext_fig
        path_sars_write = path_to_outs + dir_csv + sars_record_filename + t + ext_csv
        post_process.write_trajectories(trajectories, path_trajectories_write, header=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS,
                                       controlled_stops=CONTROLLED_STOPS)
        post_process.write_trajectories(sars, path_sars_write)

    if len(tstamps) > 1:
        t = tstamps[-1]
        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
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
        path_dtimes_write = path_to_outs + dir_figs + dwell_time_filename + t + ext_csv
        path_dtimes_plot = path_to_outs + dir_figs + dwell_time_filename + t + ext_fig

        ltimes_mean, ltimes_std, dtimes_mean, dtimes_std = post_process.travel_times_from_trajectory_set(
            trajectories_set, IDX_DEP_T, IDX_ARR_T)

        ltimes_lbl = ['mean', 'stdev']
        ltimes_x_y_lbls = ['stops', 'seconds']

        post_process.write_link_times(ltimes_mean, ltimes_std, stops_loc, path_ltimes_write, STOPS)
        post_process.plot_link_times(ltimes_mean, ltimes_std, STOPS, path_ltimes_plot, ltimes_lbl,
                                     x_y_lbls=ltimes_x_y_lbls, controlled_stops=CONTROLLED_STOPS)
        post_process.write_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_write, STOPS)
        post_process.plot_dwell_times(dtimes_mean, dtimes_std, stops_loc, path_dtimes_plot, STOPS,
                                      controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ HEADWAY ---------------------------------------------

        path_plot_headway_combined = path_to_outs + dir_figs + headway_filename + t + ext_fig

        post_process.plot_headway(path_plot_headway_combined, headway_comb, STOPS,
                                  controlled_stops=CONTROLLED_STOPS)

        # ------------------------------------------ LOAD PROFILE ------------------------------------------
        mean_load_comb, std_load_comb, ons_comb, offs_comb = post_process.pax_per_trip_from_trajectory_set(
            trajectories_set, IDX_LOAD,
            IDX_PICK, IDX_DROP)

        path_plot_load_profile_combined = path_to_outs + dir_figs + load_profile_filename + t + ext_fig

        post_process.plot_load_profile(ons_comb, offs_comb, mean_load_comb, STOPS, l_dev=std_load_comb,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'pax per trip', 'pax load'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- DENIED BOARDINGS -----------------------------------------
        denied_boardings_comb = post_process.denied_from_trajectory_set(trajectories_set, IDX_DENIED, ons_comb)

        path_plot_denied_boardings_combined = path_to_outs + dir_figs + denied_filename + t + ext_fig

        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', '1 in 1,000 pax'],
                                       controlled_stops=CONTROLLED_STOPS)

        # ---------------------------------------- HOLDING TIME ---------------------------------------------
        path_plot_holding_time_comb = path_to_outs + dir_figs + hold_time_filename + t + ext_fig
        path_plot_hold_time_distribution = path_to_outs + dir_figs + hold_time_distribution_filename + t + ext_fig
        hold_time_comb, hold_time_all = post_process.hold_time_from_trajectory_set(trajectories_set, IDX_HOLD_TIME)
        post_process.plot_bar_chart(hold_time_comb, STOPS, path_plot_holding_time_comb,
                                    x_y_lbls=['stop id', 'seconds'], controlled_stops=CONTROLLED_STOPS)
        post_process.plot_histogram(hold_time_all, path_plot_hold_time_distribution)

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

# access_past_results('out/var', 'sars_record', '1201', 'out/txt')
