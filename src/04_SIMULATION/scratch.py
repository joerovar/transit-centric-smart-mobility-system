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
import seaborn as sns

a1 = np.array([2] * 3 + [4] * 5 + [6] * 5 + [8] * 2)
a2 = np.array([3] * 3 + [5] * 5 + [7] * 5 + [0] * 2)
sns.kdeplot(a1)
sns.kdeplot(a2)
plt.show()
