import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

a = pd.DataFrame([['ABC123', '20'], ['DEF456', '22']], columns=['ID', 'age'])
print(a)

numbers = a['ID'].str[-3:]
a.insert(1, 'ID NUMBER', numbers)
a['ID'] = a['ID'].str.slice(stop=3)
print(a)

d = {'386-388': 12, '388-340': 15}
e = pd.DataFrame(d.items(), columns=['orig_stop', 'time'])
d_stop = e['orig_stop'].str.split('-')[1]
e.insert(1, 'dest_stop', d_stop)
e['orig_stop'] = e['orig_stop'].str.split('-')[0]
print(e)