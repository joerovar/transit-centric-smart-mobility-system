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
from agents_sim import Bus
from input import BLOCK_TRIPS_INFO, BLOCK_DICT
from copy import deepcopy

buses = []
for block_trip_set in BLOCK_TRIPS_INFO:
    block_id = block_trip_set[0]
    trip_set = block_trip_set[1]
    buses.append(Bus(block_id, trip_set))

# switch bus 1 and bus 4
bus1 = buses[1]
bus4 = buses[4]
switch_idx = 3
copy1 = bus1.pending_trips
copy4 = bus4.pending_trips
id1 = bus1.bus_id
id4 = bus4.bus_id
print(buses[1].pending_trips)
print(buses[4].pending_trips)
print(buses[1].bus_id)
print(buses[4].bus_id)
bus1.bus_id = id4
bus4.bus_id = id1
bus1.pending_trips = copy4[switch_idx:]
bus4.pending_trips = copy4[:switch_idx] + copy1
print(buses[1].pending_trips)
print(buses[4].pending_trips)
print(buses[1].bus_id)
print(buses[4].bus_id)
