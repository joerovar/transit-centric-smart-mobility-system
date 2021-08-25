import output_fns
from file_paths import *


def write_results():
    trajectories = output_fns.load(path_tr_load)
    stops_loc = output_fns.get_stop_loc(path_stops_loc)
    output_fns.write_trajectories(trajectories, path_tr_csv)
    output_fns.write_link_times(trajectories, stops_loc, path_lt)
    wait_times = output_fns.load(path_wt_load)
    output_fns.write_wait_times(wait_times, stops_loc, path_wt)
    return


def plot_results():
    wait_times_actual = output_fns.load(path_wt_load)
    wait_times_from_h = output_fns.load(path_wtc_load)
    headway = output_fns.load(path_hw_load)
    output_fns.plot_stop_headway(headway, path_hw_fig)
    trajectories = output_fns.load(path_tr_load)
    output_fns.plot_trajectories(trajectories, path_tr_fig)
    lbl = ['wait time', 'wait time from headway']
    output_fns.plot_multiple_bar_charts(wait_times_actual, wait_times_from_h, path_wtc_fig, lbl)
    boardings = output_fns.load(path_bd_load)
    denied_boardings = output_fns.load(path_db_load)
    output_fns.plot_bar_chart(boardings, path_bd_fig)
    output_fns.plot_bar_chart(denied_boardings, path_db_fig)
    return
