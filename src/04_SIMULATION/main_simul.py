import output
import simulation_env
from file_paths import *
import post_process
import agents as Agents

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
        i = env.bus_idx
        trip_id = env.active_trips[i]
        bw_h = env.trips_sars[trip_id][-1][0][3]
        fw_h = env.trips_sars[trip_id][-1][0][2]
        if fw_h < bw_h:
            env.take_action(5)
        else:
            env.take_action(1)
        done = env.prep()
    if save:
        env.process_results()
        post_process.save(path_tr_save, env.trajectories)
        post_process.save(path_sars_save, env.trips_sars)


# running
run_base_drl(save=True)

# outputs
output.get_results()
output.get_rl_results()

print("ran in %.2f seconds" % (time.time()-st))
