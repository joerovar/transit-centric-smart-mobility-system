import pandas as pd
import numpy as np
from Variable_Inputs import DATES, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO, DIR_ROUTE_OUTS
from Variable_Inputs import scheduled_trajectories_out, HIGH_CAPACITY, LOW_CAPACITY, MRH_STOPS, DIR_ROUTE
from Variable_Inputs import ODT_STOP_IDS, ARR_RATES, ODT_BIN_MINS, STOPS_OUT_NAMES, KEY_STOPS_IDX
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound
from Output_Processor import validate_trip_t_outbound, trajectory_plots, cv_hw_plot, pax_times_plot, load_plots
from Output_Processor import plot_run_times, plot_pax_profile, denied_count, compare_input_pax_rates
from Simulation_Processor import run_base_dispatching, rl_dispatch
from Output_Processor import expressing_analysis
from ins.Scenarios_81 import BENCH_SCENARIOS, BENCH_METHOD_LBLS, BENCH_PRC_CANCEL, DIR_OUT_BENCH, DIR_VALIDATE
import os
from datetime import datetime

# train RL
# # rl_dispatch(1000, train=True, prob_cancel=0.25)

def test_scenarios2(replications=1, save_results=False, single_scenario=None):
    tstamp = datetime.now().strftime('%m%d-%H%M')
    if save_results:
        os.mkdir(DIR_ROUTE_OUTS + tstamp)
    df = pd.read_csv('ins/Scenarios_81.csv')
    nr_scen = df['scenario'].max()
    # first assign cancelled blocks
    prc_cancel = df['prc_cancel'].unique().tolist()
    lst_blocks_cancel = []
    for p in prc_cancel:
        tmp_cancel = []
        for j in range(len(BLOCK_TRIPS_INFO)):
                if np.random.uniform(0, 1) < p/100:
                    tmp_cancel.append(BLOCK_TRIPS_INFO[j][0])
        lst_blocks_cancel.append(tmp_cancel)
    df2 = pd.DataFrame({'prc_cancel':prc_cancel, 'blocks_cancel':lst_blocks_cancel})
    df = df.merge(df2, on='prc_cancel').sort_values(by='scenario')
    df.to_csv(DIR_ROUTE_OUTS + tstamp + '/scenarios.csv', index=False)
    if single_scenario:
        df_params = df.loc[df['scenario'] == single_scenario, ['strategy', 'capacity', 'blocks_cancel']].copy()
        strategy, capacity, blocks_cancel = df_params.values.tolist()[0]
    for n in range(1, nr_scen+1):
        df_params = df.loc[df['scenario'] == n, ['strategy', 'capacity', 'blocks_cancel']].copy()
        strategy, capacity, blocks_cancel = df_params.values.tolist()[0]
        save_folder = DIR_ROUTE_OUTS + tstamp + '/' + str(n) if save_results else None
        run_base_dispatching(replications=replications, save_results=save_results, save_folder=save_folder,
                                control_strategy=strategy, cancelled_blocks=blocks_cancel, capacity=capacity)
    return

def validate(sim_out_path=None, sim_in_path=None, avl_path=None, apc_path=None):
    sim_df_out = pd.read_pickle(sim_out_path)
    sim_df_in = pd.read_pickle(sim_in_path)
    avl_df = pd.read_csv(avl_path)
    apc_df = pd.read_csv(apc_path)
    wkday_trip_ids = np.load(DIR_ROUTE + 'wkday_schd_trip_ids_out.npy').tolist()
    validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT)
    validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
    validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES, wkday_trip_ids,
                            key_stops_idx=KEY_STOPS_IDX, stop_names=STOPS_OUT_NAMES)
    validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
                             DIR_VALIDATE + 'trip_t_dist_out.png',
                             DIR_VALIDATE + 'dwell_t_dist_out.png',
                             DATES, ignore_terminals=True)
    compare_input_pax_rates(ODT_STOP_IDS, STOPS_OUT_FULL_PATT, KEY_STOPS_IDX, ons=True)
    compare_input_pax_rates(ODT_STOP_IDS, STOPS_OUT_FULL_PATT, KEY_STOPS_IDX, ons=False)
    plot_pax_profile(sim_df_out, STOPS_OUT_FULL_PATT, path_savefig=DIR_VALIDATE + 'pax_profile.png',
                    path_savefig_single=DIR_VALIDATE + 'pax_profile_single.png', stop_names=STOPS_OUT_NAMES, 
                    key_stops_idx=KEY_STOPS_IDX, mrh_hold_stops=MRH_STOPS)
    plot_pax_profile(apc_df, STOPS_OUT_FULL_PATT, path_savefig=DIR_VALIDATE + 'pax_profile_in.png',
                     apc=True, path_savefig_single=DIR_VALIDATE +'pax_profile_single_in.png', stop_names=STOPS_OUT_NAMES,
                     key_stops_idx=KEY_STOPS_IDX, mrh_hold_stops=MRH_STOPS, trip_ids=wkday_trip_ids)
    return


def plot_bench_results(hr_start_period=6.5, hr_end_period=8.0):
    time_period = (int(hr_start_period * 60 * 60), int(hr_end_period * 60 * 60))
    fig_dir = DIR_OUT_BENCH + 'trajectories.png'
    trajectory_plots([sc[-1] for sc in BENCH_SCENARIOS], BENCH_METHOD_LBLS,
                     scheduled_trajectories_out, time_period, replication=1, fig_dir=fig_dir)

    fig_dir = DIR_OUT_BENCH + 'run_times.png'
    df_95rt = plot_run_times(BENCH_SCENARIOS, BENCH_PRC_CANCEL, BENCH_METHOD_LBLS, 
                            STOPS_OUT_FULL_PATT, fig_dir=fig_dir)

    df_db = denied_count(BENCH_SCENARIOS, time_period, BENCH_PRC_CANCEL, BENCH_METHOD_LBLS)

    fig_dir = DIR_OUT_BENCH + 'cv_hw.png'
    df_h = cv_hw_plot(BENCH_SCENARIOS, STOPS_OUT_FULL_PATT,
                      time_period, BENCH_PRC_CANCEL, BENCH_METHOD_LBLS, fig_dir=fig_dir)
    fig_dir = DIR_OUT_BENCH + 'pax_times.png'
    df_pt = pax_times_plot(BENCH_SCENARIOS, STOPS_OUT_FULL_PATT, STOPS_OUT_FULL_PATT,
                           time_period, BENCH_METHOD_LBLS, BENCH_PRC_CANCEL, fig_dir=fig_dir)
    fig_dir = DIR_OUT_BENCH + '95th_loads.png'
    df_95l = load_plots(BENCH_SCENARIOS, BENCH_PRC_CANCEL, BENCH_METHOD_LBLS, STOPS_OUT_FULL_PATT, 
                        time_period, fig_dir=fig_dir, quantile=0.95)

    fig_dir = DIR_OUT_BENCH + '50th_loads.png'
    df_50l = load_plots(BENCH_SCENARIOS, BENCH_PRC_CANCEL, BENCH_METHOD_LBLS, STOPS_OUT_FULL_PATT, 
                        time_period, fig_dir=fig_dir, quantile=0.5)

    df_results = pd.concat([df_h, df_pt, df_95l, df_50l, df_95rt, df_db], ignore_index=True)
    df_results.to_csv(DIR_OUT_BENCH + 'numer_results.csv', index=False)
    return

def analyze_expressing(avl_path=None, sim_out_path=None):
    # expected dwell time savings
    avl_df, sim_df = pd.read_csv(avl_path), pd.read_pickle(sim_out_path)
    expressing_analysis(avl_df, sim_df, STOPS_OUT_FULL_PATT, 7*60*60, 8*60*60, DATES,
                        ARR_RATES, ODT_BIN_MINS, ODT_STOP_IDS)
    return

# analyze_expressing(avl_path='ins/rt_81_2022-05/avl.csv', sim_out_path='outs/rt_81_2022-05/NC/0824-132632-trip_record_ob.pkl')
# validate(sim_out_path='outs/rt_81_2022-05/NC/0824-132632-trip_record_ob.pkl', sim_in_path='outs/rt_81_2022-05/NC/0824-132632-trip_record_ib.pkl',
#          avl_path='ins/rt_81_2022-05/avl.csv', apc_path='ins/rt_81_2022-05/avl.csv')
# test_scenarios(eds=True, prc_cancel=[15], low_capacity=True, replications=1)
# test_scenarios(nc=True, prc_cancel=[0], replications=20, save_results=True, limited_capacity=True)
# test_scenarios(eds=True, hds=True, hdsmrh=True, hdstx=True, prc_cancel=BENCH_PRC_CANCEL, replications=20,
#                low_capacity=True, save_results=True)
# plot_bench_results()
test_scenarios2(save_results=True)