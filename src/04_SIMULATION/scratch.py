import matplotlib.pyplot as plt
import pandas as pd
# from Inputs import FOCUS_TRIPS
import numpy as np


d = {'stop0': [1,2,3], 'stop_id':['386','8613','15136']}
df = pd.DataFrame(d)
df['seq_diff'] = df['stop0'].diff().shift(-1)
df['stop1'] = df['stop_id'].shift(-1)
print(df)
df = df.dropna(subset=['seq_diff'])
df = df[df['seq_diff'] == 1.0]

print(df)