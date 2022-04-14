import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


main_stop_times = pd.read_csv('in/raw/rt20_stop_times.csv')
extra_stop_times = pd.read_csv('in/raw/rt20_extra.csv')


avl_cols = ['trip_id', 'route_id', 'stop_sequence', 'avl_arr_time',
            'avl_dep_time', 'avl_arr_sec', 'avl_dep_sec', 'schd_sec', 'stop_id']
main_stop_times_avl = main_stop_times[avl_cols]
extra_stop_times_avl = extra_stop_times[avl_cols]
rt20_avl = pd.concat([main_stop_times_avl, extra_stop_times], ignore_index=True)
rt20_avl = rt20_avl.sort_values(by=['trip_id', 'stop_sequence'])
rt20_avl.to_csv('in/raw/rt20_avl.csv', index=False)
