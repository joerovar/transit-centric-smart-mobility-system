import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import ODT_START_INTERVAL, ODT_END_INTERVAL, ODT_INTERVAL_LEN_MIN, DATES
from input import STOPS_OUTBOUND
from post_process import save, load
from pre_process import bi_proportional_fitting

odt_stops = np.load('in/xtr/rt_20_odt_stops.npy')
stop_t_df = pd.read_csv('in/raw/rt20_stop_times.csv')
odt = np.load('in/xtr/rt_20_odt_stops.npy')

# arr_rates = np.zeros(shape=(48, len(odt_stops)))
# drop_rates = np.zeros(shape=(48, len(odt_stops)))
# for stop_idx in range(len(odt_stops)):
#     print(f'stop {stop_idx+1}')
#     temp_df = stop_t_df[stop_t_df['stop_id'] == int(odt_stops[stop_idx])]
#     for interval_idx in range(48):
#         t_edge0 = interval_idx * ODT_INTERVAL_LEN_MIN * 60
#         t_edge1 = (interval_idx + 1) * ODT_INTERVAL_LEN_MIN * 60
#         pax_df = temp_df[temp_df['avl_dep_sec'] % 86400 <= t_edge1]
#         pax_df = pax_df[pax_df['avl_dep_sec'] % 86400 >= t_edge0]
#         ons_rate_by_date = np.zeros(len(DATES))
#         ons_rate_by_date[:] = np.nan
#         for k in range(len(DATES)):
#             day_df = pax_df[pax_df['avl_arr_time'].astype(str).str[:10] == DATES[k]]
#             if not day_df.empty:
#                 ons_rate_by_date[k] = (day_df['ron'].sum() + day_df['fon'].sum()) * 60 / ODT_INTERVAL_LEN_MIN
#         arr_rates[interval_idx, stop_idx] = np.nanmean(ons_rate_by_date)
#         offs_rate_by_date = np.zeros(len(DATES))
#         offs_rate_by_date[:] = np.nan
#         for k in range(len(DATES)):
#             day_df = pax_df[pax_df['avl_arr_time'].astype(str).str[:10] == DATES[k]]
#             if not day_df.empty:
#                 offs_rate_by_date[k] = (day_df['roff'].sum() + day_df['foff'].sum()) * 60 / ODT_INTERVAL_LEN_MIN
#         drop_rates[interval_idx, stop_idx] = np.nanmean(offs_rate_by_date)
# save('in/xtr/apc_on_counts.pkl', arr_rates)
# save('in/xtr/apc_off_counts.pkl', drop_rates)
arr_rates = load('in/xtr/apc_on_counts.pkl')
drop_rates = load('in/xtr/apc_off_counts.pkl')
stops_lst = list(odt_stops)
outbound_idx = [stops_lst.index(int(s)) for s in STOPS_OUTBOUND]
outbound_arr_rates = arr_rates[:, outbound_idx]
outbound_drop_rates = drop_rates[:, outbound_idx]


    # balance_target_factor = np.sum(target_ons) / np.sum(target_offs)
    # balanced_target_offs = target_offs * balance_target_factor
    # for i in range(15):
    #     # balance rows
    #     actual_ons = np.nansum(od, axis=1)
    #     factor_ons = np.divide(target_ons, actual_ons, out=np.zeros_like(target_ons), where=actual_ons != 0)
    #     od = od * factor_ons[:, np.newaxis]
    #
    #     # balance columns
    #     actual_offs = np.nansum(od, axis=0)
    #     factor_offs = np.divide(balanced_target_offs, actual_offs, out=np.zeros_like(target_offs), where=actual_offs != 0)
    #     od = od * factor_offs
    #
    #     # to check for tolerance we first assign 1.0 to totals of zero which cannot be changed by the method
    #     factor_ons[actual_ons == 0] = 1.0
    #     factor_offs[actual_offs == 0] = 1.0
    # scaled_od_set = np.array(od)
