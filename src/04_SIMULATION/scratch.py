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


mypath = 'out/var'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'sars_record' in f and '1201' in f]
only_files_type2 = [f.replace('.pkl', '.csv') for f in onlyfiles]
print(only_files_type2)