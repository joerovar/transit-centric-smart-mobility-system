import pandas as pd
import numpy as np
from Variable_Inputs import DATES, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO, DIR_ROUTE_OUTS
from Variable_Inputs import scheduled_trajectories_out, MRH_STOPS, DIR_ROUTE, STOPS_IN_FULL_PATT
from Variable_Inputs import ODT_STOP_IDS, ARR_RATES, ODT_BIN_MINS, STOPS_OUT_NAMES, KEY_STOPS_IDX
from Variable_Inputs import TRIPS_OUT_INFO, LINK_TIMES_MEAN, BLOCK_IDS
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound
from Output_Processor import validate_trip_t_outbound, trajectory_plots, cv_hw_plot, pax_times_plot, load_plots
from Output_Processor import plot_run_times, plot_pax_profile, denied_count, compare_input_pax_rates
from Simulation_Processor import run_base_dispatching, rl_dispatch
from Output_Processor import expressing_analysis, check_bus_speeds
from eval import run_rl_scenario
import os
from datetime import datetime

# train RL
# # rl_dispatch(1000, train=True, prob_cancel=0.25)

def run_scenarios(replications=1, save_results=False, single_scenario=None):
    tstamp = datetime.now().strftime('%m%d-%H%M')
    if save_results:
        os.mkdir(DIR_ROUTE_OUTS + tstamp)
    df = pd.read_csv('ins/Scenarios_81.csv')
    if single_scenario:
        nr_scen = 1
        df = df[df['scenario'] == single_scenario].copy()
        p = df['p_cancel'].iloc[0]
        lst_blocks_cancel = []
        tmp_cancel = []
        if p>0.0:
            tmp_cancel = np.random.choice(BLOCK_IDS, replace=False, size=int(p*len(BLOCK_IDS))).tolist()
        lst_blocks_cancel.append(tmp_cancel)
        df['blocks_cancel'] = lst_blocks_cancel
    else:
        nr_scen = df['scenario'].max()
        # first assign cancelled blocks
        p_cancel = df['p_cancel'].unique().tolist()
        lst_blocks_cancel = []
        for p in p_cancel:
            tmp_cancel = np.random.choice(BLOCK_IDS, replace=False, size=int(p*len(BLOCK_IDS))).tolist()
            lst_blocks_cancel.append(tmp_cancel)
        df2 = pd.DataFrame({'p_cancel':p_cancel, 'blocks_cancel':lst_blocks_cancel})
        df = df.merge(df2, on='p_cancel').sort_values(by='scenario')
    if save_results:
        df.to_csv(DIR_ROUTE_OUTS + tstamp + '/scenarios.csv', index=False)
    if single_scenario:
        df_params = df.loc[df['scenario'] == single_scenario, ['strategy', 'capacity', 'blocks_cancel']].copy()
        strategy, capacity, blocks_cancel = df_params.values.tolist()[0]
        save_folder = DIR_ROUTE_OUTS + tstamp + '/' + str(single_scenario) if save_results else None
        if strategy == 'RL':
            run_rl_scenario(episodes=replications, save_results=save_results, save_folder=save_folder,
                                cancelled_blocks=blocks_cancel)
        else:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder=save_folder,
                                        control_strategy=strategy, cancelled_blocks=blocks_cancel, capacity=capacity)
    else:
        for n in range(1, nr_scen+1):
            df_params = df.loc[df['scenario'] == n, ['strategy', 'capacity', 'blocks_cancel']].copy()
            strategy, capacity, blocks_cancel = df_params.values.tolist()[0]
            save_folder = DIR_ROUTE_OUTS + tstamp + '/' + str(n) if save_results else None
            if strategy == 'RL':
                run_rl_scenario(episodes=replications, save_results=save_results, save_folder=save_folder,
                                cancelled_blocks=blocks_cancel)
            else:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder=save_folder,
                                        control_strategy=strategy, cancelled_blocks=blocks_cancel, capacity=capacity)
    return

def validate(tstamp, n_scenario, avl_path=None, apc_path=None):
    sim_df_out = pd.read_pickle(DIR_ROUTE_OUTS + tstamp + '/' + str(n_scenario) + '/trip_record_ob.pkl')
    sim_df_in = pd.read_pickle(DIR_ROUTE_OUTS + tstamp + '/' + str(n_scenario) + '/trip_record_ib.pkl')
    avl_df = pd.read_csv(avl_path)
    apc_df = pd.read_csv(apc_path)
    wkday_trip_ids = np.load(DIR_ROUTE + 'wkday_schd_trip_ids_out.npy').tolist()
    path_validate = DIR_ROUTE_OUTS + tstamp + '/validation'
    path_save = DIR_ROUTE_OUTS + tstamp + '/validation/'
    if not os.path.exists(path_validate):
        os.mkdir(path_validate)
    validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT, path_save)
    validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5, path_save, STOPS_IN_FULL_PATT)
    validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES, wkday_trip_ids,
                            path_save, key_stops_idx=KEY_STOPS_IDX, stop_names=STOPS_OUT_NAMES)
    validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
                             path_save + 'trip_t_dist_out.png',
                             path_save + 'dwell_t_dist_out.png', DATES, ignore_terminals=True)
    compare_input_pax_rates(ODT_STOP_IDS, STOPS_OUT_FULL_PATT, KEY_STOPS_IDX, path_save, ons=True)
    compare_input_pax_rates(ODT_STOP_IDS, STOPS_OUT_FULL_PATT, KEY_STOPS_IDX, path_save, ons=False)
    plot_pax_profile(sim_df_out, STOPS_OUT_FULL_PATT, path_savefig=path_save + 'pax_profile.png',
                    path_savefig_single=path_save + 'pax_profile_single.png', stop_names=STOPS_OUT_NAMES, 
                    key_stops_idx=KEY_STOPS_IDX, mrh_hold_stops=MRH_STOPS)
    plot_pax_profile(apc_df, STOPS_OUT_FULL_PATT, path_savefig=path_save + 'pax_profile_in.png',
                     apc=True, path_savefig_single=path_save +'pax_profile_single_in.png', stop_names=STOPS_OUT_NAMES,
                     key_stops_idx=KEY_STOPS_IDX, mrh_hold_stops=MRH_STOPS, trip_ids=wkday_trip_ids)
    expressing_analysis(avl_df, sim_df_out, STOPS_OUT_FULL_PATT, 7*60*60, 8*60*60, DATES,
                    ARR_RATES, ODT_BIN_MINS, ODT_STOP_IDS, path_save)
    check_bus_speeds(STOPS_OUT_FULL_PATT, TRIPS_OUT_INFO, STOPS_OUT_NAMES, KEY_STOPS_IDX, LINK_TIMES_MEAN, path_save)
    return


def plot_bench_results(tstamp, hr_start_period=6.5, hr_end_period=8.0):
    df = pd.read_csv(DIR_ROUTE_OUTS + tstamp + '/scenarios.csv')
    path_bench = DIR_ROUTE_OUTS + tstamp + '/benchmark'
    path_save = DIR_ROUTE_OUTS + tstamp + '/benchmark/'
    if not os.path.exists(path_bench):
        os.mkdir(path_bench)
    time_period = (int(hr_start_period * 60 * 60), int(hr_end_period * 60 * 60))

    fig_dir = path_save + 'trajectories.png'
    nr_cancel_max = df['nr_cancel'].max()

    trajectory_plots(tstamp, df, nr_cancel_max, scheduled_trajectories_out, time_period, 
                    replication=2, fig_dir=fig_dir)

    fig_dir = path_save + 'run_times.png'
    df_95rt = plot_run_times(tstamp, df, STOPS_OUT_FULL_PATT, fig_dir=fig_dir)

    df_db = denied_count(tstamp, df, time_period)

    fig_dir = path_save + 'cv_hw.png'
    df_h = cv_hw_plot(tstamp, df, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir)

    fig_dir = path_save + 'pax_times.png'
    df_pt = pax_times_plot(tstamp, df, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir)

    fig_dir = path_save + '95th_loads.png'
    df_95l = load_plots(tstamp, df, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.95)

    fig_dir = path_save + '50th_loads.png'
    df_50l = load_plots(tstamp, df, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.5)

    df_results = pd.concat([df_h, df_pt, df_95l, df_50l, df_95rt, df_db], ignore_index=True)
    df_results.to_csv(path_save + 'numer_results.csv', index=False)
    return


if __name__ == "__main__":
    run_scenarios(save_results=True,replications=20)
    # plot_bench_results('0907-0951')
    # validate('0907-0951', 1, avl_path='ins/rt_81_2022-05/avl.csv', apc_path='ins/rt_81_2022-05/avl.csv')