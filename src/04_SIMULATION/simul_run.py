import core
from file_paths import *
import output_fns


def run(save=False):
    env = core.SimulationEnv()
    done = env.reset_simulation()
    while not done:
        done = env.prep()
    if save:
        env.process_results()
        output_fns.save(path_wt_save, env.adjusted_wait_time)
        output_fns.save(path_tr_save, env.trajectories)
        output_fns.save(path_hw_save, env.recorded_headway)
        output_fns.save(path_bd_save, env.tot_pax_at_stop)
        output_fns.save(path_db_save, env.tot_denied_boardings)
        output_fns.save(path_wtc_save, env.wait_time_from_h)
