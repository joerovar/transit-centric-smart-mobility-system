import output
import simulation_env
from file_paths import *
import post_process
from datetime import datetime

st = time.time()


def run_base(episodes=1, save=False, plot=False):
    tstamps = []
    for i in range(episodes):
        env = simulation_env.SimulationEnv()
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-5])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
    if plot:
        output.get_results(tstamps)


def run_base_control(episodes=1, save=False, plot=False):
    tstamps = []
    for i in range(episodes):
        env = simulation_env.SimulationEnvWithControl()
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-5])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
    if plot:
        output.get_base_control_results(tstamps)


run_base_control(episodes=6, save=True, plot=True)

print("ran in %.2f seconds" % (time.time()-st))
