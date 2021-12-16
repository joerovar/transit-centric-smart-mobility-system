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


def bus_availability(sched_arr, sched_dep, actual_arr, iden):
    actual_dep = [0] * len(sched_dep)
    for n in range(len(sched_dep)):
        actual_dep[n] = max(sched_dep[n], actual_arr[n])
    plt.plot(sched_arr, np.arange(1, len(sched_arr)+1),'--', label='scheduled arrivals', color='green', alpha=0.5)
    plt.plot(sched_dep, np.arange(1, len(sched_dep) + 1),'--', label='scheduled departures', color='red', alpha=0.5)
    plt.plot(actual_arr, np.arange(1, len(actual_arr) + 1),label='actual arrivals', color='green')
    plt.plot(actual_dep, np.arange(1, len(actual_dep) + 1),label='actual departures', color='red')
    plt.legend()
    plt.savefig('in/vis/sample_bus_availability' + iden + '.png')

    plt.close()
    actual_dep_hw = [j - k for j, k in zip(actual_dep[1:], actual_dep[:-1])]
    return actual_dep_hw

# extract


# for generation of graphs you want:
dep_delay1_params = load('in/xtr/rt_20-2019-09/dep_delay1_params.pkl')
# print(dep_delay1_params)
trip_time1_params = load('in/xtr/rt_20-2019-09/trip_time1_params.pkl')
dep_delay2_params = load('in/xtr/rt_20-2019-09/dep_delay2_params.pkl')
# print(dep_delay2_params)
trip_time2_params = load('in/xtr/rt_20-2019-09/trip_time2_params.pkl')
sched_deps_out1 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound1.pkl')
sched_deps_out2 = load('in/xtr/rt_20-2019-09/scheduled_departures_outbound2.pkl')

sim_arr_hws = []
sched_arr_out = load('in/xtr/rt_20-2019-09/scheduled_arrivals_outbound.pkl')
observed_hw = load('in/xtr/rt_20-2019-09/arrival_headway_outbound.pkl')
sched_dep_in = load('in/xtr/rt_20-2019-09/departure_times_inbound.pkl')
sim_dep_hws = []
for i in range(4):
    # long pattern
    dep_delay1 = lognorm.rvs(dep_delay1_params[0], loc=dep_delay1_params[1], scale=dep_delay1_params[2], size=len(sched_deps_out1))
    trip_times1 = norm.rvs(loc=trip_time1_params[0], scale=trip_time1_params[1], size=len(sched_deps_out1))
    sim_arr1 = [i + j + k for i, j, k in zip(sched_deps_out1, dep_delay1, trip_times1)]
    # plt.hist(dep_delay1, bins=15, ec='black')
    # plt.show()
    # plt.close()
    # short pattern
    dep_delay2 = lognorm.rvs(dep_delay2_params[0], loc=dep_delay2_params[1], scale=dep_delay2_params[2], size=len(sched_deps_out2))
    trip_times2 = norm.rvs(loc=trip_time2_params[0], scale=trip_time2_params[1], size=len(sched_deps_out2))
    sim_arr2 = [i + j + k for i, j, k in zip(sched_deps_out2, dep_delay2, trip_times2)]
    # plt.hist(dep_delay2, bins=15, ec='black')
    # plt.show()
    # plt.close()
    # join and validate with scheduled arrivals and arrival headway
    sim_arr = sim_arr1 + sim_arr2
    sim_arr.sort()
    sim_arr_hw = [round(i - j) for i, j in zip(sim_arr[1:], sim_arr[:-1])]
    sim_arr_hws += sim_arr_hw
    sim_dep_hw = bus_availability(sched_arr_out, sched_dep_in, sim_arr, str(i))
    sim_dep_hws += sim_dep_hw


sns.kdeplot(observed_hw, label='observed')
sns.kdeplot(sim_arr_hws, label='simulated')
plt.xlabel('outbound arrival headway (seconds)')
plt.legend()
plt.savefig('in/vis/validate_arrival_hw_out.png')
plt.close()

dep_hw = load('in/xtr/rt_20-2019-09/departure_headway_inbound.pkl')
sns.kdeplot(dep_hw, label='observed')
sns.kdeplot(sim_dep_hws, label='simulated')
plt.xlabel('inbound departure headway (seconds)')
plt.legend()
plt.savefig('in/vis/validate_depart_hw_in.png')
plt.close()

print([np.mean(sim_dep_hws), np.median(sim_dep_hws), np.std(sim_dep_hws)])
#
# # pattern 1 (long)
# scheduled_departures1 = np.array([])
# scheduled_arrivals1 = np.array([])
# dep_delays1 = np.random.lognormal(dep_delay1_params[0], dep_delay1_params[1], size=scheduled_departures1.size)
# trip_times1 = np.random.normal(trip_time1_params[0], trip_time1_params[1], size=scheduled_departures1.size)
# actual_departures1 = scheduled_departures1 + dep_delays1
# actual_arrivals1 = actual_departures1 + trip_times1



