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

CV = 0.15
LOGN_S = np.sqrt(np.log(np.power(CV, 2) + 1))
s = LOGN_S
mean_runtime1 = 81
runtime = lognorm.rvs(s, scale=mean_runtime1, size=20)
print(runtime)
print(runtime.std())
mean_runtime2 = 30
runtime2 = lognorm.rvs(s, scale=mean_runtime2, size=20)
print(runtime2)
print(runtime2.std())