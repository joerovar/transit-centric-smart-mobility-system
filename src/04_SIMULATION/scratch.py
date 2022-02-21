import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from datetime import timedelta
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join
from scipy.stats import lognorm, norm
# from classes_simul import Passenger, Stop, Trip
import seaborn as sn
from post_process import save, load
from pre_process import remove_outliers
import os


# base_date = '2019-09-03 00:00:00'
# for df in (df3, df4, df5, df6):
#     d = df[df['route_id'] == str(20)]
#     d = d[['route_id', 'trip_id', 'stop_sequence', 'event_time', 'departure_time',
#            'dwell_time', 'passenger_load']]
#     d['base_date'] = [base_date] * len(d.index)
#     d['avl_sec'] = d['departure_time'].astype('datetime64[ns]') - d['base_date'].astype('datetime64[ns]')
#     d['avl_sec'] = d['avl_sec'].dt.total_seconds()
#     d = d.dropna(subset=['avl_sec'])
#     d['avl_sec'] = d['avl_sec'].round(decimals=1)
#     d = d.drop('base_date', axis=1)
#     df_main = df_main.append(d, ignore_index=True)


# df_main['trip_id'] = df_main['trip_id'].astype(str).str[:-2]
# df_main['trip_id'] = df_main['trip_id'].astype(int)
# df_main['route_id'] = df_main['route_id'].astype(int)
# df_main.to_csv('route20_stop_time_dwell.csv', index=False)
# df_dwell = pd.read_csv('route20_stop_time_dwell.csv')
# df_no_dwell = pd.read_csv('route20_stop_time.csv')
# df_no_dwell = df_no_dwell[['schd_trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'schd_sec', 'event_time']]
# df_main = df_dwell.merge(df_no_dwell, left_on=['trip_id', 'stop_sequence', 'event_time'],
#                          right_on=['schd_trip_id', 'stop_sequence', 'event_time'])
# df_main.to_csv('route20_stop_time_final.csv', index=False)
