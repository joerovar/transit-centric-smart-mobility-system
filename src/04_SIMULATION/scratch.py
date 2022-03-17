import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from datetime import timedelta
import random
from datetime import datetime
from post_process import save
from post_process import load

nr_scenarios = 3
nr_methods = 3
a = np.ones(shape=(nr_methods*nr_scenarios,))
print(a)
for i in range(nr_scenarios):
    print(range(nr_methods*i, nr_methods*(i+1)))
    print(a[nr_methods*i:nr_methods*(i+1)])
