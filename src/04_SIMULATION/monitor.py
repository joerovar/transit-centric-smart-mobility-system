import core
import results
import csv
import matplotlib.pyplot as plt
import pickle

# env = core.SimulationEnv()
# done = env.reset_simulation()
# while not done:
#     done = env.prep()
# results.write_csv_trajectories(env.trajectories, 'out/text/trip data.dat')
# results.plot_headway_per_stop(env.recorded_headway, 'out/figs/headways per stop.png')

with open('out/vars/trajectories_0816.pkl', 'rb') as tf:
    trajectories = pickle.load(tf)

with open('out/vars/headway_0816.pkl', 'rb') as tf:
    headway = pickle.load(tf)

results.write_link_times(trajectories)
results.write_wait_times(headway)
# link_times_mean = results.store_linktimes(env.trajectories)
# wait_times_mean = results.store_waittimes(env.recorded_headway)

