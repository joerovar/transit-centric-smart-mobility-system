import post_process
from input import *


def get_results(tstamps):
    trajectories_set = []
    for t in tstamps:

        path_trajectories_load = path_to_outs + dir_var + 'trajectories_' + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        stops_loc = post_process.get_stop_loc(path_stops_loc)
        trajectories_header = ['stop_id, arr_t, dep_t, bus_load, pick_count, drop_count, denied_count']
        path_trajectories_write = path_to_outs + dir_csv + 'trajectories_' + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + 'trajectories_' + t + ext_fig
        post_process.write_trajectories(trajectories, path_trajectories_write, label=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS)

        path_ltimes_write = path_to_outs + dir_csv + 'link_times_' + t + ext_csv
        post_process.write_link_times(trajectories, IDX_DEP_T, IDX_ARR_T, stops_loc, path_ltimes_write)

        headway, wtimes, wtimes_from_h = post_process.get_headway_from_trajectories(trajectories, IDX_PICK, IDX_DENIED)
        wait_time_lbl = ['wait time', 'wait time from headway']
        t_per_stop_lbl = ['stop id', 'seconds']
        path_plot_wtime_compare = path_to_outs + dir_figs + 'wait_time_comparison_' + t + ext_fig
        path_plot_wtime = path_to_outs + dir_figs + 'wait_time_' + t + ext_fig
        path_plot_headway = path_to_outs + dir_figs + 'headway_' + t + ext_fig
        post_process.plot_multiple_bar_charts(wtimes, wtimes_from_h, path_plot_wtime_compare, wait_time_lbl, STOPS,
                                              x_y_lbls=t_per_stop_lbl)
        post_process.plot_bar_chart(wtimes, STOPS, path_plot_wtime, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway, headway, STOPS)

        mean_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, average=True)
        stdev_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, stdev=True)
        boardings = post_process.count_from_trajectories(trajectories, IDX_PICK)
        drop_offs = post_process.count_from_trajectories(trajectories, IDX_DROP)
        denied_boardings = post_process.count_from_trajectories(trajectories, IDX_DENIED)
        path_plot_load_profile = path_to_outs + dir_figs + 'load_profile_' + t + ext_fig
        path_plot_denied_boardings = path_to_outs + dir_figs + 'denied_boardings_' + t + ext_fig
        post_process.plot_load_profile(boardings, drop_offs, mean_load, stdev_load, STOPS,
                                       pathname=path_plot_load_profile,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings, denied_boardings, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])

    if len(tstamps) > 1:
        t = tstamps[-1]
        # do the combined stuff
        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
        path_plot_wtime_combined = path_to_outs + dir_figs + 'wait_time_combined_' + t + ext_fig
        path_plot_headway_combined = path_to_outs + dir_figs + 'headway_combined_' + t + ext_fig
        t_per_stop_lbl = ['stop id', 'seconds']
        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway_combined, headway_comb, STOPS)

        mean_load_comb, std_load_comb = post_process.average_from_trajectory_set(trajectories_set, IDX_LOAD)
        boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_PICK)
        drop_offs_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DROP)
        denied_boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DENIED)
        path_plot_load_profile_combined = path_to_outs + dir_figs + 'load_profile_combined_' + t + ext_fig
        path_plot_denied_boardings_combined = path_to_outs + dir_figs + 'denied_boardings_combined_' + t + ext_fig
        post_process.plot_load_profile(boardings_comb, drop_offs_comb, mean_load_comb, std_load_comb, STOPS,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])
    return


def get_base_control_results(tstamps):
    # THINGS TO COMBINE
    trajectories_set = []
    for t in tstamps:

        path_trajectories_load = path_to_outs + dir_var + 'trajectories_' + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        stops_loc = post_process.get_stop_loc(path_stops_loc)
        trajectories_header = ['stop_id, arr_t, dep_t, bus_load, pick_count, drop_count, denied_count, hold_sec']
        path_trajectories_write = path_to_outs + dir_csv + 'trajectories_' + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + 'trajectories_' + t + ext_fig
        post_process.write_trajectories(trajectories, path_trajectories_write, label=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS)

        path_ltimes_write = path_to_outs + dir_csv + 'link_times_' + t + ext_csv
        post_process.write_link_times(trajectories, IDX_DEP_T, IDX_ARR_T, stops_loc, path_ltimes_write)

        headway, wtimes, wtimes_from_h = post_process.get_headway_from_trajectories(trajectories, IDX_PICK, IDX_DENIED)
        wait_time_lbl = ['wait time', 'wait time from headway']
        t_per_stop_lbl = ['stop id', 'seconds']
        path_plot_wtime_compare = path_to_outs + dir_figs + 'wait_time_comparison_' + t + ext_fig
        path_plot_wtime = path_to_outs + dir_figs + 'wait_time_' + t + ext_fig
        path_plot_headway = path_to_outs + dir_figs + 'headway_' + t + ext_fig
        post_process.plot_multiple_bar_charts(wtimes, wtimes_from_h, path_plot_wtime_compare, wait_time_lbl, STOPS,
                                              x_y_lbls=t_per_stop_lbl)
        post_process.plot_bar_chart(wtimes, STOPS, path_plot_wtime, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway, headway, STOPS)

        mean_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, average=True)
        stdev_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, stdev=True)
        boardings = post_process.count_from_trajectories(trajectories, IDX_PICK)
        drop_offs = post_process.count_from_trajectories(trajectories, IDX_DROP)
        denied_boardings = post_process.count_from_trajectories(trajectories, IDX_DENIED)
        path_plot_load_profile = path_to_outs + dir_figs + 'load_profile_' + t + ext_fig
        path_plot_denied_boardings = path_to_outs + dir_figs + 'denied_boardings_' + t + ext_fig
        post_process.plot_load_profile(boardings, drop_offs, mean_load, stdev_load, STOPS,
                                       pathname=path_plot_load_profile,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings, denied_boardings, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])
        holding_time = post_process.count_from_trajectories(trajectories, IDX_HOLD_TIME)
        path_plot_holding_time = path_to_outs + dir_figs + 'holding_time_' + t + ext_fig
        post_process.plot_bar_chart(holding_time, STOPS, path_plot_holding_time, x_y_lbls=['stop id', 'seconds'])

    if len(tstamps) > 1:
        t = tstamps[-1]
        # do the combined stuff
        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
        path_plot_wtime_combined = path_to_outs + dir_figs + 'wait_time_combined_' + t + ext_fig
        path_plot_headway_combined = path_to_outs + dir_figs + 'headway_combined_' + t + ext_fig
        t_per_stop_lbl = ['stop id', 'seconds']
        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway_combined, headway_comb, STOPS)

        mean_load_comb, std_load_comb = post_process.average_from_trajectory_set(trajectories_set, IDX_LOAD)
        boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_PICK)
        drop_offs_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DROP)
        denied_boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DENIED)
        holding_time_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_HOLD_TIME)
        path_plot_holding_time_comb = path_to_outs + dir_figs + 'holding_time_combined_' + t + ext_fig
        path_plot_load_profile_combined = path_to_outs + dir_figs + 'load_profile_combined_' + t + ext_fig
        path_plot_denied_boardings_combined = path_to_outs + dir_figs + 'denied_boardings_combined_' + t + ext_fig
        post_process.plot_load_profile(boardings_comb, drop_offs_comb, mean_load_comb, std_load_comb, STOPS,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])
        post_process.plot_bar_chart(holding_time_comb, STOPS, path_plot_holding_time_comb, x_y_lbls=['stop id', 'seconds'])

    return


def get_rl_results(tstamps):

    # THINGS TO COMBINE
    trajectories_set = []
    for t in tstamps:
        path_sars = path_to_outs + dir_var + 'sars_record_' + tstamps[-1] + ext_var
        sars = post_process.load(path_sars)
        path_trajectories_load = path_to_outs + dir_var + 'trajectories_' + t + ext_var
        trajectories = post_process.load(path_trajectories_load)
        if len(tstamps) > 1:
            trajectories_set.append(trajectories)

        stops_loc = post_process.get_stop_loc(path_stops_loc)
        trajectories_header = ['stop_id, arr_t, dep_t, bus_load, pick_count, drop_count, denied_count, hold_sec, skip']
        path_trajectories_write = path_to_outs + dir_csv + 'trajectories_' + t + ext_csv
        path_trajectories_plot = path_to_outs + dir_figs + 'trajectories_' + t + ext_fig
        path_sars_write = path_to_outs + dir_csv + 'sars_record_' + t + ext_csv
        post_process.write_trajectories(trajectories, path_trajectories_write, label=trajectories_header)
        post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_trajectories_plot, STOPS)
        post_process.write_trajectories(sars, path_sars_write)

        path_ltimes_write = path_to_outs + dir_csv + 'link_times_' + t + ext_csv
        post_process.write_link_times(trajectories, IDX_DEP_T, IDX_ARR_T, stops_loc, path_ltimes_write)

        headway, wtimes, wtimes_from_h = post_process.get_headway_from_trajectories(trajectories, IDX_PICK, IDX_DENIED)
        wait_time_lbl = ['wait time', 'wait time from headway']
        t_per_stop_lbl = ['stop id', 'seconds']
        path_plot_wtime_compare = path_to_outs + dir_figs + 'wait_time_comparison_' + t + ext_fig
        path_plot_wtime = path_to_outs + dir_figs + 'wait_time_' + t + ext_fig
        path_plot_headway = path_to_outs + dir_figs + 'headway_' + t + ext_fig
        post_process.plot_multiple_bar_charts(wtimes, wtimes_from_h, path_plot_wtime_compare, wait_time_lbl, STOPS,
                                              x_y_lbls=t_per_stop_lbl)
        post_process.plot_bar_chart(wtimes, STOPS, path_plot_wtime, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway, headway, STOPS)

        mean_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, average=True)
        stdev_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, stdev=True)
        boardings = post_process.count_from_trajectories(trajectories, IDX_PICK)
        drop_offs = post_process.count_from_trajectories(trajectories, IDX_DROP)
        denied_boardings = post_process.count_from_trajectories(trajectories, IDX_DENIED)
        path_plot_load_profile = path_to_outs + dir_figs + 'load_profile_' + t + ext_fig
        path_plot_denied_boardings = path_to_outs + dir_figs + 'denied_boardings_' + t + ext_fig
        post_process.plot_load_profile(boardings, drop_offs, mean_load, stdev_load, STOPS,
                                       pathname=path_plot_load_profile,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings, denied_boardings, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])
        holding_time = post_process.count_from_trajectories(trajectories, IDX_HOLD_TIME)
        path_plot_holding_time = path_to_outs + dir_figs + 'holding_time_' + t + ext_fig
        post_process.plot_bar_chart(holding_time, STOPS, path_plot_holding_time, x_y_lbls=['stop id', 'seconds'])

    if len(tstamps) > 1:
        t = tstamps[-1]
        # do the combined stuff
        headway_comb, wait_time_comb, wait_time_from_h_comb = get_headway_from_trajectory_set(trajectories_set,
                                                                                              IDX_PICK, IDX_DENIED)
        path_plot_wtime_combined = path_to_outs + dir_figs + 'wait_time_combined_' + t + ext_fig
        path_plot_headway_combined = path_to_outs + dir_figs + 'headway_combined_' + t + ext_fig
        t_per_stop_lbl = ['stop id', 'seconds']
        post_process.plot_bar_chart(wait_time_comb, STOPS, path_plot_wtime_combined, x_y_lbls=t_per_stop_lbl)
        post_process.plot_stop_headway(path_plot_headway_combined, headway_comb, STOPS)

        mean_load_comb, std_load_comb = post_process.average_from_trajectory_set(trajectories_set, IDX_LOAD)
        boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_PICK)
        drop_offs_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DROP)
        denied_boardings_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_DENIED)
        holding_time_comb = post_process.count_from_trajectory_set(trajectories_set, IDX_HOLD_TIME)
        path_plot_holding_time_comb = path_to_outs + dir_figs + 'holding_time_combined_' + t + ext_fig
        path_plot_load_profile_combined = path_to_outs + dir_figs + 'load_profile_combined_' + t + ext_fig
        path_plot_denied_boardings_combined = path_to_outs + dir_figs + 'denied_boardings_combined_' + t + ext_fig
        post_process.plot_load_profile(boardings_comb, drop_offs_comb, mean_load_comb, std_load_comb, STOPS,
                                       pathname=path_plot_load_profile_combined,
                                       x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
        post_process.plot_pax_per_stop(path_plot_denied_boardings_combined, denied_boardings_comb, STOPS,
                                       x_y_lbls=['stop id', 'denied boardings (pax)'])
        post_process.plot_bar_chart(holding_time_comb, STOPS, path_plot_holding_time_comb, x_y_lbls=['stop id', 'seconds'])
    return

