import core
import results
import csv
import matplotlib.pyplot as plt
import pickle
import time

path_stops_loc = 'in/gtfs/stops.txt'

tstamp_save = time.strftime("%m%d-%H%M")
path_tr_save = 'out/trajectories_' + tstamp_save + '.pkl'
path_hw_save = 'out/headway_' + tstamp_save + '.pkl'

tstamp_load = tstamp_save
path_tr_load = 'out/trajectories_' + tstamp_load + '.pkl'
path_hw_load = 'out/headway_' + tstamp_load + '.pkl'
path_lt = 'out/link_times_' + tstamp_load + '.csv'
path_wt = 'out/stop_wait_times_' + tstamp_load + '.csv'
path_hw_fig = 'out/headway_' + tstamp_load + '.png'
path_tr_csv = 'out/trajectories_' + tstamp_load + '.csv'

env = core.SimulationEnv()
done = env.reset_simulation()
while not done:
    done = env.prep()

results.save(path_tr_save, env.trajectories)
results.save(path_hw_save, env.recorded_headway)

trajectories = results.load(path_tr_load)
headway = results.load(path_hw_load)

results.write_trajectories(trajectories, path_tr_csv)
results.plot_stop_headway(headway, path_hw_fig)
stops_loc = results.get_stop_loc(path_stops_loc)
results.write_link_times(trajectories, stops_loc, path_lt)
results.write_wait_times(headway, stops_loc, path_wt)


