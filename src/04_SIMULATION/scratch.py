import matplotlib.pyplot as plt
import pandas as pd
# from Inputs import FOCUS_TRIPS
import numpy as np


d = {'stop_id': ['2', '5'], 'stop_seq': [1, 2]}
df = pd.DataFrame(d)
s0 = df[df['stop_seq'] == 1]
if not s0.empty:
    print(s0['stop_id'].iloc[0])
