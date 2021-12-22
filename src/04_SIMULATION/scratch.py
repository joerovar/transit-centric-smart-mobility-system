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

li = [[1, 2, 3, [10, 20, 30]], [2, 5, 4, [11, 22, 33]]]
found = 0
for i in range(len(li)):
    if li[i][3][0] == 11:
        found = i
        break
print(found)
