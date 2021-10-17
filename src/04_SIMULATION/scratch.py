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


arr_times = np.array([0, 10, 20, 30, 40])
dep_times = np.array([2, 12, 22, 32, 42])
times = np.vstack((arr_times, dep_times))
times = times.flatten(order='F')
print(times)

