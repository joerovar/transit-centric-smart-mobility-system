import output
from simulation_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl
from file_paths import *
import post_process
from datetime import datetime

st = time.time()


def run_base_detailed(episodes=2, save=False, plot=False, time_dep_tt=True, time_dep_dem=True):
    tstamps = []
    for i in range(episodes):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-4])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
            path_completed_pax = path_to_outs + dir_var + 'completed_pax_' + tstamps[-1] + ext_var
            post_process.save(path_completed_pax, env.completed_pax)
    if plot:
        output.get_results(tstamps)
    return


def run_base_control_detailed(episodes=2, save=False, plot=False, time_dep_tt=True, time_dep_dem=True):
    tstamps = []
    for i in range(episodes):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-4])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
            path_completed_pax = path_to_outs + dir_var + 'completed_pax_' + tstamps[-1] + ext_var
            post_process.save(path_completed_pax, env.completed_pax)
    if plot:
        output.get_base_control_results(tstamps)
    return


run_base_detailed(episodes=10, save=True, plot=True)
# run_base_control_detailed(episodes=10, save=True, plot=True)
output.benchmark_comparisons()
print("ran in %.2f seconds" % (time.time()-st))
