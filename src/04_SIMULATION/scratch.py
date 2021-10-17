import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from datetime import timedelta
import random


class Person:
    def __init__(self, age, os):
        self.age = age
        self.os = os

    def report(self):
        print(f'my dude is {self.age} years old and uses {self.os}')


class TechGeek(Person):
    def __init__(self, name, *args, **kwargs):
        super(TechGeek, self).__init__(*args, **kwargs)
        self.name = name

    def report(self):
        print(f'This tech geek called {self.name} is {self.age} years old and uses {self.os}')


p = pd.DataFrame([['90', 1, 30], ['90', 2, 45], ['90', 3, 66], ['91', 1, 145], ['91', 2, 162], ['91', 3, 190]],
                 columns=['trip_id', 'stop_seq', 'dep_time'])

trips = p['trip_id'].unique()
fig, ax = plt.subplots()
p.reset_index().groupby(['trip_id']).plot(x='stop_seq', y='dep_time', ax=ax, marker='*')
start, end = ax.get_xlim()
plt.xticks([1, 2, 3])
plt.legend(trips)
plt.show()
