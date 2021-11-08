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


stop = ['386', '388', '390']
load = [12, 22, 35]
d = {'stop': stop,
     'load': load}
df = pd.DataFrame(d)
print(df.columns)