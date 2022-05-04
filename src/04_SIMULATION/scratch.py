import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import ODT_START_INTERVAL, ODT_END_INTERVAL, ODT_INTERVAL_LEN_MIN, DATES
# from input import STOPS_OUTBOUND
from post_process import save, load
from pre_process import bi_proportional_fitting

a = np.array([-5, 3, 9])
print(list(a[(a>0) & (a<8)]))
