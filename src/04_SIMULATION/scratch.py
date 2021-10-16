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
# [100.  36. 181. 135.  35.  37.  21. 152.  30.  26. 318. 135. 296.]
# while do_continue == 'y':
#     s = input('seconds: ')
#     conversion = timedelta(seconds=float(s))
#     converted_time = str(conversion)
#     print(converted_time)
#     do_continue = input('another one? ')

p = pd.DataFrame([[1, 0.2], [3, 4]], columns=['a', 'b'])
i = p.index[p['a'] == 3]
print(p)
p.drop(p[p['a'] < p['b']])
print(p)


