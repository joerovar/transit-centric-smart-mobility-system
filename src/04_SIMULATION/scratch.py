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

a = [[1, 3], [2, 4]]
for j in range(len(a)):
    a[j] = [i+1 for i in a[j]]
print(a)