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
from classes_simul import Passenger, Stop, Trip

journey_times = {'o': ['386', '386','388'], 'd': ['388', '388','389'], 'jt': [30, 45, 15]}
jt_df = pd.DataFrame(journey_times)
jt_mean = jt_df[(jt_df['o'] == '386') & (jt_df['d'] == '388')]['jt'].mean()
print(jt_mean)
