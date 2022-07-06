import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Inputs import DATES, START_TIME_SEC, END_TIME_SEC, LINK_TIMES_MEAN, STOPS_OUT_FULL_PATT, BLOCK_TRIPS_INFO
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound, \
    validate_trip_t_outbound
from Input_Processor import get_interval
from Simulation_Processor import run_base, train_rl, test_rl, run_base_dispatching, rl_dispatch

nr_reps = 5

# run_base_dispatching(nr_reps, save_results=True, save_folder='NC_dispatch')
# run_base_dispatching(nr_reps, save_results=True, control_type='EH', save_folder='EH_dispatch')
# rl_dispatch(nr_reps, save_results=True, save_folder='RL_dispatch', tstamp_policy='0629-1713')
# prob_cancel = 0.1
# cancelled_blocks = [[] for _ in range(nr_reps)]
# for i in range(nr_reps):
#     for j in range(len(BLOCK_TRIPS_INFO)):
#         if np.random.uniform(0, 1) < prob_cancel:
#             cancelled_blocks[i].append(BLOCK_TRIPS_INFO[j][0])
#
# run_base_dispatching(nr_reps, save_results=True, save_folder='NC_dispatch',
#                      prob_cancel=prob_cancel, cancelled_blocks=cancelled_blocks)
# run_base_dispatching(nr_reps, save_results=True, control_type='EH', save_folder='EH_dispatch',
#                      prob_cancel=prob_cancel, cancelled_blocks=cancelled_blocks)
# rl_dispatch(nr_reps, save_results=True, save_folder='RL_dispatch', prob_cancel=prob_cancel,
#             cancelled_blocks=cancelled_blocks, tstamp_policy='0629-1713')
# prob_cancel = 0.2
# cancelled_blocks = [[] for _ in range(nr_reps)]
# for i in range(nr_reps):
#     for j in range(len(BLOCK_TRIPS_INFO)):
#         if np.random.uniform(0, 1) < prob_cancel:
#             cancelled_blocks[i].append(BLOCK_TRIPS_INFO[j][0])
#
# run_base_dispatching(replications=nr_reps, save_results=True, save_folder='NC_dispatch',
#                      prob_cancel=prob_cancel, cancelled_blocks=cancelled_blocks)
# run_base_dispatching(replications=nr_reps, save_results=True, control_type='EH', save_folder='EH_dispatch',
#                      prob_cancel=prob_cancel, cancelled_blocks=cancelled_blocks)
# rl_dispatch(nr_reps, save_results=True, save_folder='RL_dispatch', prob_cancel=prob_cancel,
#             cancelled_blocks=cancelled_blocks, tstamp_policy='0629-1713')
# rl_dispatch(600, train=True, prob_cancel=0.2, weight_hold_t=1/1000)

# run_base_dispatching(replications=25, prob_cancel=0.2, save_results=False)
# sim_df_out = pd.read_pickle('out/NC/0524-153428-trip_record_outbound.pkl')

# run_base(replications=10, control_eh=True, save_results=True)
# sim_df_out = pd.read_pickle('out/NC_Terminal/0627-134924-trip_record_outbound.pkl')
# sim_df_in = pd.read_pickle('out/NC_Terminal/0627-134924-trip_record_inbound.pkl')
# pax_df = pd.read_pickle('out/NC_Terminal/0627-134924-pax_record.pkl')

# validate ----------------
# avl_df = pd.read_csv('in/raw/rt20_avl_2019-09.csv')
# validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC)
# validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
# validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUT_FULL_PATT, DATES)
# validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUT_FULL_PATT,
#                          'out/compare/validate/trip_t_dist_out.png', 'out/compare/validate/dwell_t_dist_out.png', DATES,
#                          ignore_terminals=True)

# results (get for all intervals in simulation) -------------------
# wait times
# RBT
# headway
# trip time distribution
# load profile

# nc_pax_df = pd.read_pickle('out/NC/0524-153428-pax_record.pkl')
# eh_pax_df = pd.read_pickle('out/EH/0601-142625-pax_record.pkl')
#
# interval0 = get_interval(START_TIME_SEC, 60)
# interval1 = get_interval(END_TIME_SEC, 60)
#
# wt_per_interval_nc = [[] for _ in range(interval0, interval1)]
# wt_per_interval_eh = [[] for _ in range(interval0, interval1)]
#
# for interval in range(interval0, interval1):
#     # wait time
#     temp_df_nc = nc_pax_df[(nc_pax_df['arr_time'] >= interval*60*60) & (nc_pax_df['arr_time'] <= (interval+1)*60*60)].copy()
#     temp_df_eh = eh_pax_df[
#         (eh_pax_df['arr_time'] >= interval * 60 * 60) & (eh_pax_df['arr_time'] <= (interval + 1) * 60 * 60)].copy()
#     temp_df_nc['wait_time'] = temp_df_nc['board_time'] - temp_df_nc['arr_time']
#     temp_df_eh['wait_time'] = temp_df_eh['board_time'] - temp_df_eh['arr_time']
#     wt_per_interval_nc[interval-interval0] = round(temp_df_nc['wait_time'].mean()/60, 2)
#     wt_per_interval_eh[interval-interval0] = round(temp_df_eh['wait_time'].mean()/60, 2)
