import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

a = pd.DataFrame([['386', '388', 3.2], ['388', '386', 4.2]], columns=['stop1', 'stop2', 'time'])
print(a)

b = pd.DataFrame([['386', 33.1, -88.2], ['388', 34.1, -85.0]], columns=['stop', 'stop_lon', 'stop_lat'])
b = b.rename(columns={'stop': 'stop1'})
c = pd.merge(a, b, on='stop1')
c = c.rename(columns={'stop_lon': 'stop1_lon', 'stop_lat': 'stop1_lat'})
b = b.rename(columns={'stop1': 'stop2'})
c = pd.merge(c, b, on='stop2')
c = c.rename(columns={'stop_lon': 'stop2_lon', 'stop_lat': 'stop2_lat'})
print(c)