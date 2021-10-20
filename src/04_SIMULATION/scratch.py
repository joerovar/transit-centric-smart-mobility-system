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
        self.age_in_10_yrs = 0

    def report(self):
        self.age_in_10_yrs = self.age
        self.age += 1
        print(f'my dude is {self.age} years old and uses {self.os} and he will be {self.age_in_10_yrs} in 10 yrs')


class TechGeek(Person):
    def __init__(self, name, *args, **kwargs):
        super(TechGeek, self).__init__(*args, **kwargs)
        self.name = name

    def report(self):
        print(f'This tech geek called {self.name} is {self.age} years old and uses {self.os}')


sars = [[1, 2], 4, 5, [1,3]]
s, a, r, s_ = sars
print(s, a, r, s)