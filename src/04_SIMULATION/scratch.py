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


df = pd.read_csv('in/raw/odt_for_opt.csv')
stops = ['386', '388', '389']
input_start_interval = 84
start_interval = 14
end_interval = 20
input_end_interval = 120
step_intervals = 6
input_interval_groups = []
for i in range(input_start_interval, input_end_interval, step_intervals):
    input_interval_groups.append([j for j in range(i, i + step_intervals)])
od = np.zeros(shape=(end_interval-start_interval, len(stops), len(stops)))
for i in range(len(stops)):
    for j in range(i+1, len(stops)):
        temp_df = df[df['BOARDING_STOP'] == float(stops[i])]
        temp_df = temp_df[temp_df['INFERRED_ALIGHTING_GTFS_STOP'] == float(stops[i+1])]
        for g in input_interval_groups:
            pax_df = temp_df[temp_df['bin_5'].isin(g)]
            time_idx = input_interval_groups.index(g)
            pax = pax_df['mean'].sum()
            od[time_idx, i, j] = pax

print(od)