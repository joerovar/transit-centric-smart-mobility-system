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
import seaborn as sns

x = load('in/xtr/rt_20-2019-09/cv_headway_inbound.pkl')
y = load('in/xtr/rt_20-2019-09/trip_times_inbound.pkl')
t_out = load('out/compare/benchmark/trip_time_sim.pkl')
fig, ax = plt.subplots(2)
ax[0].plot(x)
sns.histplot([i/60 for i in y], ax=ax[1], kde=True)
sns.histplot([t/60 for t in t_out], ax=ax[1], kde=True)
plt.show()
