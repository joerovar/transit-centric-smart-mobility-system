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
import post_process
from scipy.stats import lognorm, norm
# from classes_simul import Passenger, Stop, Trip
import seaborn as sn
from post_process import *

df = pd.read_csv('in/vis/trajectories_inbound.csv')
unique_trips = df['trip_id'].unique().tolist()
df_main = pd.read_csv('in/raw/route20_stop_time_merged.csv')
stops = load('in/xtr/rt_20-2019-09/route_stops.pkl')
dates = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']


on_off_counts = []
for trip in unique_trips:
    for d in dates:
        trip_df = df[df['trip_id'] == trip]
        trip_df = trip_df[trip_df['avl_arr_time'].astype(str).str[:10] == d]
        if not trip_df.empty:
            schd_sec = trip_df['schd_sec'].min()
            peak_load = trip_df['passenger_load'].max()
            on_counts = trip_df['ron'].sum() + trip_df['fon'].sum()
            off_counts = trip_df['roff'].sum() + trip_df['foff'].sum()
            on_off_counts.append((trip, d, on_counts, off_counts, round(on_counts/off_counts), str(timedelta(seconds=schd_sec)), peak_load, schd_sec))
counts_df = pd.DataFrame(on_off_counts, columns=['trip_id', 'date', 'ons', 'offs', 'proportion', 'schd_dep', 'peak_load', 'schd_sec'])
diff = counts_df['ons'].sum() / counts_df['offs'].sum()
print(diff)
# counts_df.to_csv('visualize_counts.csv', index=False)
