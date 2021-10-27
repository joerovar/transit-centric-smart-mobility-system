import post_process
from input import *


def get_results():
    trajectories = post_process.load(path_tr_load)
    stops_loc = post_process.get_stop_loc(path_stops_loc)
    trajectories_lbls = ['stop_id, arr_t, dep_t, bus_load, pick_count, drop_count, denied_count, hold_time, skip (1/0)']
    post_process.write_trajectories(trajectories, path_tr_csv, label=trajectories_lbls)
    post_process.write_link_times(trajectories, IDX_DEP_T, IDX_ARR_T, stops_loc, path_lt)

    headway, wtimes, wtimes_from_h = post_process.get_headway_from_trajectories(trajectories, IDX_PICK, IDX_DENIED)

    mean_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, average=True)
    stdev_load = post_process.count_from_trajectories(trajectories, IDX_LOAD, stdev=True)
    boardings = post_process.count_from_trajectories(trajectories, IDX_PICK)
    drop_offs = post_process.count_from_trajectories(trajectories, IDX_DROP)
    denied_boardings = post_process.count_from_trajectories(trajectories, IDX_DENIED)
    wait_time_lbl = ['wait time', 'wait time from headway']
    t_per_stop_lbl = ['stop id', 'seconds']
    post_process.plot_multiple_bar_charts(wtimes, wtimes_from_h, path_wtc_fig_, wait_time_lbl, STOPS, x_y_lbls=t_per_stop_lbl)
    post_process.plot_bar_chart(wtimes, STOPS, path_wt_fig, x_y_lbls=t_per_stop_lbl)
    post_process.plot_stop_headway(path_hw_fig, headway, STOPS)
    post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_tr_fig, STOPS)
    post_process.plot_load_profile(boardings, drop_offs, mean_load, stdev_load, STOPS, pathname=path_lp_fig, x_y_lbls=['stop id', 'total # pax', 'avg bus load', 'load stdev'])
    post_process.plot_pax_per_stop(path_db_fig, denied_boardings, STOPS, x_y_lbls=['stop id', 'denied boardings (pax)'])
    return


def get_even_headway_results():
    trajectories = post_process.load(path_tr_load)
    holding_time = post_process.count_from_trajectories(trajectories, IDX_HOLD_TIME)
    post_process.plot_bar_chart(holding_time, STOPS, path_hold_time_fig, x_y_lbls=['stop id', 'seconds'])
    return


def get_rl_results():
    trajectories = post_process.load(path_tr_load)
    sars = post_process.load(path_sars_load)
    post_process.write_trajectories(sars, path_sars_csv)
    holding_time = post_process.count_from_trajectories(trajectories, IDX_HOLD_TIME)
    post_process.plot_bar_chart(holding_time, STOPS, path_hold_time_fig, x_y_lbls=['stop id', 'seconds'])
    return


def combine_episodes():
    dates = ['0906-1906', '0906-1908', '0906-1909', '0906-1910']
    hw = []
    for d in dates:
        hw.append(post_process.load(path_to_outs + 'headway_' + d + '.pkl'))
    hw = post_process.merge_dictionaries(hw[0], hw[1], hw[2], hw[3])
    post_process.plot_stop_headway(hw, path_to_outs + 'headway_0906_merged.png', STOPS, y_scale=[-50, 1800])
    return


