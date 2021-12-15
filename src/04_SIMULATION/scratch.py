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
from scipy.stats import lognorm
# from classes_simul import Passenger, Stop, Trip
import seaborn as sns

# extract

# for generation of graphs you want:
# dep_delay1_params = (0, 0)
# trip_time1_params = (1, 1)
# dep_delay2_params = (0, 0)
# trip_time2_params = (1, 1)
#
# # pattern 1 (long)
# scheduled_departures1 = np.array([])
# scheduled_arrivals1 = np.array([])
# dep_delays1 = np.random.lognormal(dep_delay1_params[0], dep_delay1_params[1], size=scheduled_departures1.size)
# trip_times1 = np.random.normal(trip_time1_params[0], trip_time1_params[1], size=scheduled_departures1.size)
# actual_departures1 = scheduled_departures1 + dep_delays1
# actual_arrivals1 = actual_departures1 + trip_times1



