import output
import simulation_env
from file_paths import *
import post_process

st = time.time()


def run_base(save=False):
    env = simulation_env.SimulationEnv()
    done = env.reset_simulation()
    while not done:
        done = env.prep()
    if save:
        env.process_results()
        post_process.save(path_tr_save, env.trajectories)


def run_base_drl(save=False):
    env = simulation_env.SimulationEnvDeepRL()
    done = env.reset_simulation()
    done = env.prep()
    while not done:
        env.take_action(1)
        done = env.prep()
    if save:
        env.process_results()
        post_process.save(path_tr_save, env.trajectories)
        post_process.save(path_sars_save, env.trips_sars)


# running
# simulation_run.run_base(save=False)
run_base_drl(save=True)

# outputs
output.get_results()
output.get_rl_results()

print("ran in %.2f seconds" % (time.time()-st))
