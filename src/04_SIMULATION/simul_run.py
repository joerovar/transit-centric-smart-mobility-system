import simulation_env
from file_paths import *
import post_process


def run(save=False):
    env = simulation_env.SimulationEnv()
    done = env.reset_simulation()
    while not done:
        done = env.prep()
    if save:
        env.process_results()
        post_process.save(path_wt_save, env.adjusted_wait_time)
        post_process.save(path_tr_save, env.trajectories)
        post_process.save(path_wtc_save, env.wait_time_from_h)
