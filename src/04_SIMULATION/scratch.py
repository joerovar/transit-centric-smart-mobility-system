import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from datetime import datetime
import random


def get_interval(t, len_i):
    interval = int(t/(len_i*60))
    return interval


