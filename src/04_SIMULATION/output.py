import post_process
from input import *


def get_results():
    trajectories = post_process.load(path_tr_load)
    stops_loc = post_process.get_stop_loc(path_stops_loc)
    post_process.write_trajectories(trajectories, path_tr_csv)
    post_process.write_link_times(trajectories, IDX_DEP_T, IDX_ARR_T, stops_loc, path_lt)

    headway, wtimes_, wtimes_from_h = post_process.get_headway_from_trajectories(trajectories, IDX_PICK, IDX_DENIED)
    lbl = ['wait time', 'wait time from headway']

    loads = post_process.count_from_trajectories(trajectories, IDX_LOAD, average=True)
    boardings = post_process.count_from_trajectories(trajectories, IDX_PICK)
    drop_offs = post_process.count_from_trajectories(trajectories, IDX_DROP)
    denied_boardings = post_process.count_from_trajectories(trajectories, IDX_DENIED)

    post_process.plot_multiple_bar_charts(wtimes_, wtimes_from_h, path_wtc_fig_, lbl, STOPS, x_y_lbls=['stop id', 'seconds'])
    post_process.plot_bar_chart(wtimes_, STOPS, path_wt_fig, x_y_lbls=['stop id', 'seconds'])
    post_process.plot_stop_headway(path_hw_fig, headway, STOPS)
    post_process.plot_trajectories(trajectories, IDX_ARR_T, IDX_DEP_T, path_tr_fig, STOPS)
    post_process.plot_load_profile(boardings, drop_offs, loads, STOPS, pathname=path_lp_fig, x_y_lbls=['stop id', 'total # pax', 'avg bus load'])
    post_process.plot_pax_per_stop(path_db_fig, denied_boardings, STOPS, x_y_lbls=['stop id', 'denied boardings (pax)'])
    return


def get_rl_results():
    sars = post_process.load(path_sars_load)
    post_process.write_trajectories(sars, path_sars_csv)
    return

def combine_episodes():
    dates = ['0906-1906', '0906-1908', '0906-1909', '0906-1910']
    hw = []
    for d in dates:
        hw.append(post_process.load(path_to_outs + 'headway_' + d + '.pkl'))
    hw = post_process.merge_dictionaries(hw[0], hw[1], hw[2], hw[3])
    post_process.plot_stop_headway(hw, path_to_outs + 'headway_0906_merged.png', STOPS, y_scale=[-50, 1800])
    return


