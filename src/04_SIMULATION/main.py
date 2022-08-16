import pandas as pd
import numpy as np
from Variable_Inputs import DATES, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO, DIR_ROUTE_OUTS
from Variable_Inputs import scheduled_trajectories_out, HIGH_CAPACITY, LOW_CAPACITY
from Variable_Inputs import ODT_STOP_IDS
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound
from Output_Processor import validate_trip_t_outbound, trajectory_plots, cv_hw_plot, pax_times_plot, load_plots
from Output_Processor import plot_run_times, plot_pax_profile, denied_count, compare_input_ons
from Simulation_Processor import run_base_dispatching, rl_dispatch
from Output_Processor import dwell_t_outbound


# train RL
# # rl_dispatch(1000, train=True, prob_cancel=0.25)
def test_scenarios(nc=False, ds=False, dsmrh=False, dsx=False, dsxmrh=False, rl=False,
                   prob_cancel=None, rl_policy=None, replications=None, save_results=False,
                   full_capacity=False, limited_capacity=False):
    for p in prob_cancel:
        cancelled_blocks = [[] for _ in range(replications)]
        for i in range(replications):
            for j in range(len(BLOCK_TRIPS_INFO)):
                if np.random.uniform(0, 1) < p:
                    cancelled_blocks[i].append(BLOCK_TRIPS_INFO[j][0])
        if full_capacity:
            if nc:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='NC',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, capacity=HIGH_CAPACITY)
            if ds:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DS',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DS',
                                     capacity=HIGH_CAPACITY)
            if dsmrh:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DS+MRH',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DS+MRH',
                                     capacity=HIGH_CAPACITY)
            if dsx:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DSX',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DSX',
                                     capacity=HIGH_CAPACITY)
            if dsxmrh:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DSX+MRH',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DSX+MRH',
                                     capacity=HIGH_CAPACITY)
            if rl:
                rl_dispatch(replications, save_results=save_results, save_folder='RL', prob_cancel=p,
                            cancelled_blocks=cancelled_blocks, tstamp_policy=rl_policy)
        if limited_capacity:
            if nc:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='NC',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, capacity=LOW_CAPACITY)
            if ds:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DS',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DS',
                                     capacity=LOW_CAPACITY)
            if dsmrh:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DS+MRH',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DS+MRH',
                                     capacity=LOW_CAPACITY)
            if dsx:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DSX',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DSX',
                                     capacity=LOW_CAPACITY)
            if dsxmrh:
                run_base_dispatching(replications=replications, save_results=save_results, save_folder='DSX+MRH',
                                     prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DSX+MRH',
                                     capacity=LOW_CAPACITY)
            if rl:
                rl_dispatch(replications, save_results=save_results, save_folder='RL', prob_cancel=p,
                            cancelled_blocks=cancelled_blocks, tstamp_policy=rl_policy)

    return


def validate(sim_out_path=None, sim_in_path=None, avl_path=None, apc_path=None):
    sim_df_out = pd.read_pickle(sim_out_path)
    sim_df_in = pd.read_pickle(sim_in_path)
    avl_df = pd.read_csv(avl_path)
    apc_df = pd.read_csv(apc_path)

    validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT)
    validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
    validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES)
    validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
                             DIR_ROUTE_OUTS + 'compare/validate/trip_t_dist_out.png',
                             DIR_ROUTE_OUTS + 'compare/validate/dwell_t_dist_out.png',
                             DATES, ignore_terminals=True)
    compare_input_ons(ODT_STOP_IDS, STOPS_OUT_FULL_PATT)
    plot_pax_profile(sim_df_out, STOPS_OUT_FULL_PATT, path_savefig=DIR_ROUTE_OUTS + 'compare/validate/pax_profile.png')
    plot_pax_profile(apc_df, STOPS_OUT_FULL_PATT, path_savefig=DIR_ROUTE_OUTS + 'compare/validate/pax_profile_in.png',
                     apc=True)
    return


def plot_results():
    # scenarios = [['NC/0729-011225', 'NC/0729-011407', 'NC/0729-011546'],
    #              ['DS/0729-011235', 'DS/0729-011417', 'DS/0729-011555'],
    #              ['DS+MRH/0729-011245', 'DS+MRH/0729-011427', 'DS+MRH/0729-011604'],
    #              ['DSX/0729-011256', 'DSX/0729-011437', 'DSX/0729-011614'],
    #              ['DSX+MRH/0729-011306', 'DSX+MRH/0729-011447', 'DSX+MRH/0729-011623']]
    scenarios = [['NC/0811-114655', 'NC/0811-114711', 'NC/0811-114725'],
                 ['DS/0811-114658', 'DS/0811-114713', 'DS/0811-114728'],
                 ['DS+MRH/0811-114701', 'DS+MRH/0811-114716', 'DS+MRH/0811-114730'],
                 ['DSX/0811-114704', 'DSX/0811-114719', 'DSX/0811-114733'],
                 ['DSX+MRH/0811-114707', 'DSX+MRH/0811-114722', 'DSX+MRH/0811-114736']]
    method_tags = ['NC', 'DS', 'DS+MRH', 'DSX', 'DSX+MRH']
    scenario_tags = [0, 12, 25]
    replication = 3
    time_period = (int(6.5 * 60 * 60), int(8.5 * 60 * 60))
    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/trajectories.png'
    trajectory_plots([sc[-1] for sc in scenarios], method_tags,
                     scheduled_trajectories_out, time_period, replication, fig_dir=fig_dir)

    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/run_times.png'
    df_95rt = plot_run_times(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, fig_dir=fig_dir)

    df_db = denied_count(scenarios, time_period, scenario_tags, method_tags)

    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/cv_hw.png'
    df_h = cv_hw_plot(scenarios, STOPS_OUT_FULL_PATT,
                      time_period, scenario_tags, method_tags, fig_dir=fig_dir)
    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/pax_times.png'
    df_pt = pax_times_plot(scenarios, STOPS_OUT_FULL_PATT, STOPS_OUT_FULL_PATT,
                           time_period, method_tags, scenario_tags, fig_dir=fig_dir)
    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/95th_loads.png'
    df_95l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.95)

    fig_dir = DIR_ROUTE_OUTS + 'compare/benchmark/50th_loads.png'
    df_50l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.5)

    df_results = pd.concat([df_h, df_pt, df_95l, df_50l, df_95rt, df_db], ignore_index=True)
    df_results.to_csv(DIR_ROUTE_OUTS + 'compare/benchmark/numer_results.csv', index=False)
    return


def plot_results2():
    scenarios = [['NC/0729-011407', 'NC/0729-011546'],
                 ['NC/0729-011316', 'NC/0729-011457'],
                 ['DS+MRH/0729-011427', 'DS+MRH/0729-011604'],
                 ['DS+MRH/0729-011337', 'DS+MRH/0729-011517'],
                 ['DSX+MRH/0729-011447', 'DSX+MRH/0729-011623'],
                 ['DSX+MRH/0729-011357', 'DSX+MRH/0729-011536']]
    method_tags = ['NC53', 'NC80', 'DS+MRH53', 'DS+MRH80', 'DSX+MRH53', 'DSX+MRH80']
    scenario_tags = [12, 25]
    replication = 3
    time_period = (int(6.5 * 60 * 60), int(8.5 * 60 * 60))
    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/trajectories.png'
    trajectory_plots([sc[-1] for sc in scenarios], method_tags,
                     scheduled_trajectories_out, time_period, replication, fig_dir=fig_dir)

    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/run_times.png'
    df_95rt = plot_run_times(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, fig_dir=fig_dir)

    df_db = denied_count(scenarios, time_period, scenario_tags, method_tags)

    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/cv_hw.png'
    df_h = cv_hw_plot(scenarios, STOPS_OUT_FULL_PATT,
                      time_period, scenario_tags, method_tags, fig_dir=fig_dir)
    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/pax_times.png'
    df_pt = pax_times_plot(scenarios, STOPS_OUT_FULL_PATT, STOPS_OUT_FULL_PATT,
                           time_period, method_tags, scenario_tags, fig_dir=fig_dir)
    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/95th_loads.png'
    df_95l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.95)

    fig_dir = DIR_ROUTE_OUTS + 'compare/infinite capacity/50th_loads.png'
    df_50l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir, quantile=0.5)

    df_results = pd.concat([df_h, df_pt, df_95l, df_50l, df_95rt, df_db], ignore_index=True)
    df_results.to_csv(DIR_ROUTE_OUTS + 'compare/infinite capacity/numer_results.csv', index=False)
    return

def analyze_expressing():
    # expected dwell time savings
    
    # avl_df = pd.read_csv('ins/rt_81_2022-05/avl.csv')
    # sim_df = pd.read_pickle('outs/rt_81_2022-05/NC/0809-095540-trip_record_ob.pkl')
    # dwell1 = dwell_t_outbound(avl_df, 2, 10, STOPS_OUT_FULL_PATT, 'arr_sec', 'dep_sec', 60, START_TIME_SEC,
    #                           END_TIME_SEC, is_avl=True, dates=DATES)
    # dwell2 = dwell_t_outbound(sim_df, 2, 10, STOPS_OUT_FULL_PATT, 'arr_sec', 'dep_sec', 60, START_TIME_SEC, END_TIME_SEC)
    # print([np.round(np.nanmean(d)) for d in dwell1])
    # print([np.round(np.mean(d)) for d in dwell2])

    # expected left behind at all times
    return

# validate(sim_out_path='outs/rt_81_2022-05/NC/0815-213424-trip_record_ob.pkl', sim_in_path='outs/rt_81_2022-05/NC/0815-213424-trip_record_ib.pkl',
#          avl_path='ins/rt_81_2022-05/avl.csv', apc_path='ins/rt_81_2022-05/avl.csv')
# test_scenarios(nc=True, prob_cancel=[0.0], replications=15, save_results=True, limited_capacity=True)
# test_scenarios(nc=True, ds=True, dsmrh=True, dsx=True, dsxmrh=True, prob_cancel=[0.0, 0.125, 0.25], replications=15,
#                limited_capacity=True, save_results=True)
# plot_results()
# plot_results2()
