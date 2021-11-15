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

t = [1, 2, 3, 4, 5]
c = [(i,j) for i,j in zip(t,t[1:])]
print(c)