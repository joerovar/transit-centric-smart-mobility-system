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

stops = load('in/xtr/rt_20-2019-09/route_stops.pkl')
links = [(i, j) for i, j in zip(stops[:-1], stops[1:])]
df_nc = pd.read_csv('out/trajectories0.csv')
df_eh = pd.read_csv('out/trajectories1.csv')
df_rl = pd.read_csv('out/trajectories2.csv')
nr_replications = 25
header = ['stop', 'mean1', 'std1', 'mean2', 'std2', 'mean3', 'std3']
header_cv = ['stop', 'cv1', 'mean2', 'std2', 'mean3', 'std3']


def dwell_times():
    dwell_df_rows = []
    for s in stops:
        dwell_df_row = [int(s)]
        for df in (df_nc, df_eh, df_rl):
            dwell_time_means = []
            dwell_times_sd = []
            for n in range(1, nr_replications+1):
                rep_df = df[df['replication'] == n]
                stop_dwell_times = rep_df[rep_df['stop_id'] == int(s)]['dwell_sec'].to_numpy()
                dwell_time_means.append(stop_dwell_times.mean())
                dwell_times_sd.append(stop_dwell_times.std())
            dwell_df_row.append(np.around(np.mean(dwell_time_means), decimals=1))
            dwell_df_row.append(np.around(np.mean(dwell_times_sd), decimals=1))
        dwell_df_rows.append(dwell_df_row)
    dwell_df = pd.DataFrame(dwell_df_rows, columns=header)
    dwell_df.to_csv('out/dwell_times.csv', index=False)


def link_times():
    link_times_df_rows = []
    for li in links:
        link_df_row = [str(li[0]) + '-' + str(li[1])]
        for df in (df_nc, df_eh, df_rl):
            link_time_means = []
            link_times_sd = []
            for n in range(1, nr_replications+1):
                rep_df = df[df['replication'] == n]
                dep_sec = rep_df[rep_df['stop_id'] == int(li[0])]['dep_sec'].values
                arr_sec = rep_df[rep_df['stop_id'] == int(li[1])]['arr_sec'].values
                li_times = arr_sec - dep_sec
                link_time_means.append(li_times.mean())
                link_times_sd.append(li_times.std())
            link_df_row.append(np.around(np.mean(link_time_means), decimals=1))
            link_df_row.append(np.around(np.mean(link_times_sd), decimals=1))
        link_times_df_rows.append(link_df_row)
    link_times_df = pd.DataFrame(link_times_df_rows, columns=header)
    link_times_df.to_csv('out/link_times.csv', index=False)


def headway():
    headway_df_rows = []
    for s in stops:
        hw_df_row = [int(s)]
        for df in (df_nc, df_eh, df_rl):
            headway_means = []
            headway_sd = []
            for n in range(1, nr_replications+1):
                rep_df = df[df['replication'] == n]
                stop_times = rep_df[rep_df['stop_id'] == int(s)]['arr_sec'].to_list()
                stop_hws = [i-j for i, j in zip(stop_times[1:], stop_times[:-1])]
                headway_means.append(np.around(np.mean(stop_hws), decimals=1))
                headway_sd.append(np.around(np.std(stop_hws), decimals=1))
            hw_df_row.append(np.around(np.mean(headway_means), decimals=1))
            hw_df_row.append(np.around(np.mean(headway_sd), decimals=1))
        headway_df_rows.append(hw_df_row)
    headway_df = pd.DataFrame(headway_df_rows, columns=header)
    headway_df.to_csv('out/headway.csv', index=False)


def error_headway():
    errors = []
    for df in (df_nc, df_eh, df_rl):
        sd_hw_reps = []
        for n in range(1, nr_replications + 1):
            all_hw = []
            rep_df = df[df['replication'] == n]
            for s in stops:
                stop_times = rep_df[rep_df['stop_id'] == int(s)]['arr_sec'].to_list()
                stop_hws = [i-j for i, j in zip(stop_times[1:], stop_times[:-1])]
                all_hw += stop_hws
            sd_hw_reps.append(np.std(all_hw))
        print(sd_hw_reps)
        errors.append(100 * np.std(sd_hw_reps) / np.mean(sd_hw_reps))
    print(errors)
# dwell_times()
# link_times()
# headway()
# error_headway()


