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

lp = load('in/xtr/rt_20-2019-09/load_profile.pkl')
plt.plot(np.arange(len(lp)),lp)
plt.show()
