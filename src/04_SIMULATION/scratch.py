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
odt = np.load('in/xtr/rt_20_odt_rates_30.npy')

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
#         all_nan = True not in np.isfinite(ons_rate_by_date)
#         if not all_nan:
#             arr_rates[interval_idx, stop_idx] = np.nanmean(ons_rate_by_date)
#         offs_rate_by_date = np.zeros(len(DATES))
#         offs_rate_by_date[:] = np.nan
#         for k in range(len(DATES)):
#             day_df = pax_df[pax_df['avl_arr_time'].astype(str).str[:10] == DATES[k]]
#             if not day_df.empty:
#                 offs_rate_by_date[k] = (day_df['roff'].sum() + day_df['foff'].sum()) * 60 / ODT_INTERVAL_LEN_MIN
#         all_nan = True not in np.isfinite(offs_rate_by_date)
#         if not all_nan:
#             drop_rates[interval_idx, stop_idx] = np.nanmean(offs_rate_by_date)
# save('in/xtr/apc_on_counts.pkl', arr_rates)
# save('in/xtr/apc_off_counts.pkl', drop_rates)

apc_ons = load('in/xtr/apc_on_counts.pkl')
apc_offs = load('in/xtr/apc_off_counts.pkl')
stops_lst = list(odt_stops)
outbound_idx = [stops_lst.index(int(s)) for s in STOPS_OUTBOUND]
outbound_apc_ons = apc_ons[:, outbound_idx]
outbound_apc_offs = apc_offs[:, outbound_idx]

shifted_odt = np.concatenate((odt[-6:], odt[:-6]), axis=0)
scaled_odt = np.concatenate((odt[-6:], odt[:-6]), axis=0)


for i in range(shifted_odt.shape[0]):
    print(f'interval {i}')
    scaled_odt[i] = bi_proportional_fitting(shifted_odt[i], apc_ons[i], apc_offs[i])


np.save('in/xtr/rt_20_odt_rates_30_scaled.npy', scaled_odt)

scaled_odt = np.load('in/xtr/rt_20_odt_rates_30_scaled.npy')
scaled_arr_rates = np.sum(scaled_odt, axis=-1)
scaled_out_arr_rates = scaled_arr_rates[:, outbound_idx]

arr_rates = np.sum(shifted_odt, axis=-1)
out_arr_rates = arr_rates[:, outbound_idx]

print(np.sum(outbound_apc_ons, axis=-1))
print(np.sum(scaled_out_arr_rates, axis=-1))
print(np.sum(out_arr_rates, axis=-1))


