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

stops = [1, 2, 3, 4, 5]
pairs = []
for i in range(len(stops)):
    for j in range(i+1,len(stops)):
        pairs.append((i,j))
print(pairs)