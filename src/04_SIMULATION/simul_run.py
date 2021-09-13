import core
from file_paths import *
import data_tools


def run(save=False):
    env = core.SimulationEnv()
    done = env.reset_simulation()
    while not done:
        done = env.prep()
    if save:
        env.process_results()
        data_tools.save(path_wt_save, env.adjusted_wait_time)
        data_tools.save(path_tr_save, env.trajectories)
        data_tools.save(path_hw_save, env.recorded_headway)
        data_tools.save(path_bd_save, env.tot_pax_at_stop)
        data_tools.save(path_db_save, env.tot_denied_boardings)
        data_tools.save(path_wtc_save, env.wait_time_from_h)
