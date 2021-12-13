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
import datetime
from scipy.stats import lognorm
# from classes_simul import Passenger, Stop, Trip
import seaborn as sns

link = {'1-2': 23, '1-3': 33}
stdev = {'1-2': 3, '1-3': 5}
df2 = pd.DataFrame(list(link.items()), columns=['link', 'time'])
df3 = pd.DataFrame(list(stdev.items()), columns=['link', 'std'])
df_main = df2.merge(df3, on='link')

print(df_main)
