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


class Bus:
    def __init__(self, bus_id, active_trip=None, next_event_time=None):
        self.active_trip = active_trip
        self.bus_id = bus_id
        self.next_event_time = next_event_time


buses = [Bus(i, i + 10, i*10) for i in range(10)]
buses.append(Bus(10))
buses.append(Bus(11, 11+10, 0))

active_buses = [bus for bus in buses if bus.active_trip]

next_bus = min(active_buses, key=lambda bus: bus.next_event_time)
print(next_bus)

next_event_times = [bus.next_event_time for bus in active_buses]
min_event_time = min(next_event_times)
min_event_time_idxs = [i for i, x in enumerate(next_event_times) if x == min_event_time]
print(min_event_time_idxs)
for i in min_event_time_idxs:
    print(active_buses[i])
print(buses[0])
print(buses[-1])

