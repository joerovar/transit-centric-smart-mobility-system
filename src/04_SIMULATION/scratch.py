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

a = {'name': ['Joe', 'Haris', 'Milad', 'David'], 'grade': [100, 45, 99, 77]}
a_df = pd.DataFrame(a)

students_top = len(a_df[a_df['grade'] > 50])
print(students_top)
