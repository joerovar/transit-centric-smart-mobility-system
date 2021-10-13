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


# mkbhd = TechGeek('Marques', 29, 'iOS')
# mkbhd.report()

lst = [1, [1,2], 3, [5]]
print(lst)


# while do_continue == 'y':
#     s = input('seconds: ')
#     conversion = timedelta(seconds=float(s))
#     converted_time = str(conversion)
#     print(converted_time)
#     do_continue = input('another one? ')

