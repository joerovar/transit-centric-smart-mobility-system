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

# with open('out/vars/trajectories_0816.pkl', 'rb') as tf:
#     trajectories = pickle.load(tf)
#
# with open('out/vars/headway_0816.pkl', 'rb') as tf:
#     headway = pickle.load(tf)
#
# stop_location = results.get_stop_gps('in/gtfs/stops.txt')
# results.write_link_times(trajectories, stop_location, 'out/vars/link_travel_times_0816.csv')
# results.write_wait_times(headway, stop_location, 'out/vars/stop_wait_times_0816.csv')

# for storing use the same pattern as loading

