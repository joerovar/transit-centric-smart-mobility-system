import data_tools
from input import *


def write_results():
    trajectories = data_tools.load(path_tr_load)
    stops_loc = data_tools.get_stop_loc(path_stops_loc)
    data_tools.write_trajectories(trajectories, path_tr_csv)
    data_tools.write_link_times(trajectories, stops_loc, path_lt)
    wait_times = data_tools.load(path_wt_load)
    data_tools.write_wait_times(wait_times, stops_loc, path_wt)
    return


def plot_results():
    wait_times_actual = data_tools.load(path_wt_load)
    wait_times_from_h = data_tools.load(path_wtc_load)
    trajectories = data_tools.load(path_tr_load)
    headway = data_tools.get_headway_from_trajectories(trajectories)
    data_tools.plot_stop_headway(path_hw_fig, headway, STOPS)
    data_tools.plot_trajectories(trajectories, path_tr_fig, STOPS)
    lbl = ['wait time', 'wait time from headway']
    data_tools.plot_multiple_bar_charts(wait_times_actual, wait_times_from_h, path_wtc_fig, lbl, x_y_lbls=['stop id', 'seconds'])
    data_tools.plot_bar_chart(wait_times_actual, path_wt_fig, x_y_lbls=['stop id', 'seconds'])
    boardings = data_tools.count_from_trajectories(trajectories, 3)
    denied_boardings = data_tools.count_from_trajectories(trajectories, 5)
    data_tools.plot_pax_per_stop(path_bd_fig, boardings, STOPS, x_y_lbls=['stop id', 'boardings (pax)'])
    data_tools.plot_pax_per_stop(path_db_fig, denied_boardings, STOPS, x_y_lbls=['stop id', 'denied boardings (pax)'])
    return


def combine_episodes():
    dates = ['0906-1906', '0906-1908', '0906-1909', '0906-1910']
    hw = []
    for d in dates:
        hw.append(data_tools.load(path_to_outs + 'headway_' + d + '.pkl'))
    hw = data_tools.merge_dictionaries(hw[0], hw[1], hw[2], hw[3])
    data_tools.plot_stop_headway(hw, path_to_outs + 'headway_0906_merged.png', STOPS, y_scale=[-50, 1800])
    return


# def change_trajectories():
#     tstamps = ['0909-1334', '0909-1341', '0909-1343', '0909-1351']
#     for ts in tstamps:
#         tr = data_tools.load(path_to_outs + 'trajectories_' + ts + '.pkl')
#         tr = data_tools.chop_trajectories(tr, 28800, 30600)
#         data_tools.plot_trajectories(tr, path_to_outs + 'mini_trajectories_' + ts + '.png')
#     return
