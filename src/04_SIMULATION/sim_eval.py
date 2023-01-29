import pandas as pd
from Variable_Inputs import KEY_STOPS_IDX, STOPS_OUT_FULL_PATT
import numpy as np

key_stop_ids = [STOPS_OUT_FULL_PATT[ki] for ki in KEY_STOPS_IDX]

key_times = [pd.to_datetime('2022-10-16 06:00'),
pd.to_datetime('2022-10-16 09:00'),
pd.to_datetime('2022-10-16 14:00'),
pd.to_datetime('2022-10-16 17:00')]
tstamp_results = '1016-2340'

scenarios = pd.read_csv(f'outs/rt_81_2022-05_full/{tstamp_results}/scenarios.csv')['scenario'].tolist()
scenario_cv_results = []
scenario_wt_results = []
for s in scenarios:
    trip_df = pd.read_pickle(f'outs/rt_81_2022-05_full/{tstamp_results}/{s}/trip_record_ob.pkl')
    pax_df = pd.read_pickle(f'outs/rt_81_2022-05_full/{tstamp_results}/{s}/pax_record_ob.pkl')
    trip_df['arr_dt'] = pd.to_datetime('2022-10-16')+pd.to_timedelta(trip_df['arr_sec'],unit='S')
    pax_df['arr_dt'] = pd.to_datetime('2022-10-16')+pd.to_timedelta(pax_df['arr_time'],unit='S')
    cv_hws = [[] for _ in key_times[:-1]]
    wts = [[] for _ in key_times[:-1]]

    for i in range(1,21):
        sub_df = trip_df[trip_df['replication']==i].copy()
        sub_p_df = pax_df[pax_df['replication']==i].copy()
        for t in range(len(key_times)-1):
            t_df = sub_df[(sub_df['arr_dt']>=key_times[t]) & (sub_df['arr_dt']<=key_times[t+1]) & ((sub_df['stop_id']==key_stop_ids[3]) |
                        (sub_df['stop_id']==key_stop_ids[6]))].copy()
            hw = t_df['arr_dt'].diff().dt.total_seconds()
            cv = hw.std()/hw.mean()
            cv_hws[t].append(cv)

            t_df = sub_p_df[(sub_p_df['arr_dt']>=key_times[t]) & (sub_p_df['arr_dt']<=key_times[t+1])].copy()
            wt = (t_df['board_time'] - t_df['arr_time']).mean()
            wts[t].append(wt)
    
    cv_hw_results = [np.mean(c) for c in cv_hws]
    wt_results = [np.mean(w) for w in wts]
    scenario_cv_results.append(cv_hw_results)
    scenario_wt_results.append(wt_results)

df_cv = pd.DataFrame({'AM': [round(sr[0],2) for sr in scenario_cv_results],
                        'mid': [round(sr[1],2) for sr in scenario_cv_results],
                        'PM': [round(sr[2],2) for sr in scenario_cv_results]})
df_wt = pd.DataFrame({'AM': [round(sr[0]/60,2) for sr in scenario_wt_results],
                        'mid': [round(sr[1]/60,2) for sr in scenario_wt_results],
                        'PM': [round(sr[2]/60,2) for sr in scenario_wt_results]})
df_cv.to_csv(f'outs/rt_81_2022-05_full/{tstamp_results}/cv_headways.csv', index=False)
df_wt.to_csv(f'outs/rt_81_2022-05_full/{tstamp_results}/wait_times.csv', index=False)
