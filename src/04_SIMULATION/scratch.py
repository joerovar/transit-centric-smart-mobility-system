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

d = {'trip_id': [911, 526, 333], 'avl_sec': [12, 25, 33]}
df = pd.DataFrame.from_dict(d)

trip_ids = [911, 333]
df = df[df['trip_id'].isin(trip_ids)]
print(df)
