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


start = 110


a = {'10': [['386', 100], ['388', 108], ['400', 140]], '20': [['386', 105], ['388', 150], ['400', 180]]}
for k in a:
    for d in a[k]:
        if d[1] < start:
            a[k].pop(0)
        else:
            break

print(a)
