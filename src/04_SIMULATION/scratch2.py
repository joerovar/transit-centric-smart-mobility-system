import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import DATES, START_TIME_SEC, END_TIME_SEC
from input import STOPS_OUTBOUND
from pre_process import get_interval, remove_outliers
from post_process import plot_calib_hist, validate_delay_outbound, validate_delay_inbound, validate_cv_hw_outbound, validate_trip_t_outbound, trip_t_outbound

sim_df_out = pd.read_pickle('out/NC/0524-153428-trip_record_outbound.pkl')
avl_df = pd.read_csv('in/raw/rt20_avl.csv')
sim_df_in = pd.read_pickle('out/NC/0524-153428-trip_record_inbound.pkl')

validate_delay_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC)
validate_delay_inbound(avl_df, sim_df_in, START_TIME_SEC, END_TIME_SEC, 5)
validate_cv_hw_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, 60, STOPS_OUTBOUND, DATES)
validate_trip_t_outbound(avl_df, sim_df_out, START_TIME_SEC, END_TIME_SEC, STOPS_OUTBOUND,
                         'out/compare/validate/trip_t_dist_out.png', 'out/compare/validate/dwell_t_dist_out.png',
                         DATES)
