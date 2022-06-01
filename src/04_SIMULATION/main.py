import pandas as pd
from Inputs import STOPS_OUTBOUND, DATES, START_TIME_SEC, END_TIME_SEC
from Output_Processor import validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound, validate_trip_t_outbound
from Input_Processor import get_interval
from Simulation_Processor import run_base, train_rl, test_rl

sim_df_out = pd.read_pickle('out/NC/0524-153428-trip_record_outbound.pkl')
sim_df_in = pd.read_pickle('out/NC/0524-153428-trip_record_inbound.pkl')
pax_df = pd.read_pickle('out/NC/0524-153428-pax_record.pkl')

# validate ----------------
# avl_df = pd.read_csv('in/raw/rt20_avl.csv')
# validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC)
# validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
# validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUTBOUND, DATES)
# validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUTBOUND,
#                          'out/compare/validate/trip_t_dist_out.png', 'out/compare/validate/dwell_t_dist_out.png', DATES)

# results (get for all intervals in simulation) -------------------
# wait times
wt_mean = []
interval0 = get_interval(START_TIME_SEC, 60)
interval1 = get_interval(END_TIME_SEC, 60)
# RBT
# headway
# trip time distribution
# load profile

