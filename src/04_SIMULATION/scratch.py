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

df = pd.read_csv('out/trajectories1.csv')
df = df[df['stop_id'] == 386]
df = df[df['replication'] == 1]
print(df[''])

