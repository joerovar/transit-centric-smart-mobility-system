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

arr = np.zeros(shape=(2, 4, 4))
arr[:] = np.nan
arr[0, 0, 1] = 3
arr[1, 0, 1] = 0
arr[1, 2, 3] = 5
arr[0, 2, 3] = 0
means = np.nanmean(arr, axis=0)


