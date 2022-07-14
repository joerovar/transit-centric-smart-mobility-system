import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Inputs import DATES, START_TIME_SEC, END_TIME_SEC, LINK_TIMES_MEAN, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO
from Inputs import scheduled_trajectories_out
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound
from Output_Processor import validate_trip_t_outbound, trajectory_plots, cv_hw_plot, wait_time_plot, load_plots
from Simulation_Processor import run_base, train_rl, test_rl, run_base_dispatching, rl_dispatch

# nr_reps = 1

# train RL
# rl_dispatch(1000, train=True, prob_cancel=0.25)


def test_scenarios(nc=False, eh=False, ehx=False, rl=False, prob_cancel=None, rl_policy=None, replications=None,
                   save_results=False):
    for p in prob_cancel:
        cancelled_blocks = [[] for _ in range(replications)]
        for i in range(replications):
            for j in range(len(BLOCK_TRIPS_INFO)):
                if np.random.uniform(0, 1) < p:
                    cancelled_blocks[i].append(BLOCK_TRIPS_INFO[j][0])
        if nc:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='NC_dispatch',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks)
        if eh:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='EH_dispatch',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='EH')
        if rl:
            rl_dispatch(replications, save_results=save_results, save_folder='RL_dispatch', prob_cancel=p,
                        cancelled_blocks=cancelled_blocks, tstamp_policy=rl_policy)
        if ehx:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='EHX_dispatch',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='EHX')
    return


def validate(sim_out_path=None, sim_in_path=None, avl_path=None):
    sim_df_out = pd.read_pickle(sim_out_path)
    sim_df_in = pd.read_pickle(sim_in_path)
    avl_df = pd.read_csv(avl_path)

    validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC)
    validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
    validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES)
    validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
                             'out/compare/validate/trip_t_dist_out.png', 'out/compare/validate/dwell_t_dist_out.png', DATES,
                             ignore_terminals=True)
    return


def plot_results():
    scenario_titles = ['No Control', 'Even Headway']
    scenarios = [['NC_dispatch/0711-131434', 'NC_dispatch/0711-131451', 'NC_dispatch/0711-131507'],
                 ['EH_dispatch/0711-131442', 'EH_dispatch/0711-131459', 'EH_dispatch/0711-131514']]
    method_tags = ['NC', 'EH']
    scenario_tags = [0, 10, 25]
    replication = 1
    time_period = (int(6.5 * 60 * 60), int(8.5 * 60 * 60))
    fig_dir = 'out/compare/benchmark/trajectories.png'
    trajectory_plots([scenarios[0][-1], scenarios[1][-1]], scenario_titles,
                     scheduled_trajectories_out, time_period, replication, fig_dir=fig_dir)

    fig_dir = 'out/compare/benchmark/cv_hw.png'
    df_h = cv_hw_plot(scenarios, STOPS_OUT_FULL_PATT, time_period, scenario_tags, method_tags, fig_dir=fig_dir)
    fig_dir = 'out/compare/benchmark/wt.png'
    df_wt = wait_time_plot(scenarios, STOPS_OUT_FULL_PATT, time_period, method_tags, scenario_tags, fig_dir=fig_dir)
    fig_dir = 'out/compare/benchmark/loads.png'
    df_l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir)
    df_results = pd.concat([df_h, df_wt, df_l], ignore_index=True)
    df_results.to_csv('out/compare/benchmark/numer_results.csv', index=False)
    return


test_scenarios(ehx=True, prob_cancel=[0.15], replications=1)
# test_scenarios(rl=True, prob_cancel=[0.25], replications=2, rl_policy='0710-2347')
# test_scenarios(nc=True, eh=True, prob_cancel=[0.0, 0.1, 0.25], save_results=True, replications=15)
# plot_results()
