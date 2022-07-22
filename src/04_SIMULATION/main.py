import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Inputs import DATES, START_TIME_SEC, END_TIME_SEC, LINK_TIMES_MEAN, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO
from Inputs import scheduled_trajectories_out
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound
from Output_Processor import validate_trip_t_outbound, trajectory_plots, cv_hw_plot, pax_times_plot, load_plots
from Output_Processor import plot_run_times
from Simulation_Processor import run_base, train_rl, test_rl, run_base_dispatching, rl_dispatch
from Output_Processor import dwell_t_outbound

# avl_df = pd.read_csv('in/raw/rt20_avl_2019-09.csv')
# sim_df = pd.read_pickle('out/NC/0719-145052-trip_record_ob.pkl')
# dwell1 = dwell_t_outbound(avl_df, 2, 9, STOPS_OUT_FULL_PATT, 'avl_arr_sec', 'avl_dep_sec', 60, START_TIME_SEC,
#                           END_TIME_SEC, is_avl=True, dates=DATES)
# dwell2 = dwell_t_outbound(sim_df, 2, 9, STOPS_OUT_FULL_PATT, 'arr_sec', 'dep_sec', 60, START_TIME_SEC, END_TIME_SEC)
# print([np.round(np.mean(d)) for d in dwell1])
# print([np.round(np.mean(d)) for d in dwell2])

# train RL
# rl_dispatch(1000, train=True, prob_cancel=0.25)


def test_scenarios(nc=False, dc=False, dcmrh=False, dcx=False, dcxmrh=False, rl=False,
                   prob_cancel=None, rl_policy=None, replications=None, save_results=False):
    for p in prob_cancel:
        cancelled_blocks = [[] for _ in range(replications)]
        for i in range(replications):
            for j in range(len(BLOCK_TRIPS_INFO)):
                if np.random.uniform(0, 1) < p:
                    cancelled_blocks[i].append(BLOCK_TRIPS_INFO[j][0])
        if nc:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='NC',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks)
        if dc:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='DC',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DC')
        if dcmrh:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='DC+MRH',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DC+MRH')
        if dcx:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='DCX',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DCX')
        if dcxmrh:
            run_base_dispatching(replications=replications, save_results=save_results, save_folder='DCX+MRH',
                                 prob_cancel=p, cancelled_blocks=cancelled_blocks, control_strategy='DCX+MRH')
        if rl:
            rl_dispatch(replications, save_results=save_results, save_folder='RL', prob_cancel=p,
                        cancelled_blocks=cancelled_blocks, tstamp_policy=rl_policy)

    return


def validate(sim_out_path=None, sim_in_path=None, avl_path=None):
    sim_df_out = pd.read_pickle(sim_out_path)
    sim_df_in = pd.read_pickle(sim_in_path)
    avl_df = pd.read_csv(avl_path)

    validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC)
    validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
    validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES)
    validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
                             'out/compare/validate/trip_t_dist_out.png', 'out/compare/validate/dwell_t_dist_out.png',
                             DATES, ignore_terminals=True)
    return


def plot_results():
    # scenario_titles = ['NC', 'DC', 'DC+MHR',
    #                   'Dispatching Control with Expressing']
    scenarios = [['NC/0719-145052', 'NC/0719-145130', 'NC/0719-145208'],
                 ['DC/0719-145059', 'DC/0719-145137', 'DC/0719-145217'],
                 ['DC+MRH/0719-145107', 'DC+MRH/0719-145146', 'DC+MRH/0719-145239'],
                 ['DCX/0719-145115', 'DCX/0719-145153', 'DCX/0719-145247'],
                 ['DCX+MRH/0719-145122', 'DCX+MRH/0719-145201', 'DCX+MRH/0719-145255']]
    method_tags = ['NC', 'DC', 'DC+MRH', 'DCX', 'DCX+MRH']
    scenario_tags = [0, 15, 30]
    replication = 3
    time_period = (int(6.5 * 60 * 60), int(8.5 * 60 * 60))
    fig_dir = 'out/compare/benchmark/trajectories.png'
    trajectory_plots([sc[-1] for sc in scenarios], method_tags,
                     scheduled_trajectories_out, time_period, replication, fig_dir=fig_dir)

    fig_dir = 'out/compare/benchmark/cv_hw.png'
    df_h = cv_hw_plot(scenarios, STOPS_OUT_FULL_PATT,
                      time_period, scenario_tags, method_tags, fig_dir=fig_dir)
    fig_dir = 'out/compare/benchmark/wt.png'
    df_pt = pax_times_plot(scenarios, STOPS_OUT_FULL_PATT, STOPS_OUT_FULL_PATT,
                           time_period, method_tags, scenario_tags, fig_dir=fig_dir)
    fig_dir = 'out/compare/benchmark/loads.png'
    df_l = load_plots(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, time_period, fig_dir=fig_dir)

    fig_dir = 'out/compare/benchmark/run_times.png'
    plot_run_times(scenarios, scenario_tags, method_tags, STOPS_OUT_FULL_PATT, fig_dir=fig_dir)

    df_results = pd.concat([df_h, df_pt, df_l], ignore_index=True)
    df_results.to_csv('out/compare/benchmark/numer_results.csv', index=False)
    return

# validate(sim_out_path='out/NC/0719-145052-trip_record_ob.pkl', sim_in_path='out/NC/0719-145052-trip_record_ib.pkl',
#         avl_path='in/raw/rt20_avl_2019-09.csv')
# test_scenarios(nc=True, prob_cancel=[0.0], replications=10, save_results=True)
# test_scenarios(dcmrh=True, prob_cancel=[0.2], replications=1)
# test_scenarios(rl=True, prob_cancel=[0.25], replications=2, rl_policy='0710-2347')
# test_scenarios(nc=True, dc=True, dcmrh=True, dcx=True, dcxmrh=True,
#                prob_cancel=[0.0, 0.15, 0.3], save_results=True, replications=15)
# plot_results()
