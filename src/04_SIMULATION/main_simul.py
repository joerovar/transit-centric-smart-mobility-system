import output
import simulation_env
from file_paths import *
import post_process
from datetime import datetime
import agents as Agents

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


# def run_base_control(save=False):
#     env = simulation_env.SimulationEnvWithControl()
#     done = env.reset_simulation()
#     while not done:
#         done = env.prep()
#     if save:
#         env.process_results()
#         post_process.save(path_tr_save, env.trajectories)
#
#
# def run_base_drl(save=False):
#     env = simulation_env.SimulationEnvDeepRL()
#     done = env.reset_simulation()
#     done = env.prep()
#     while not done:
#         i = env.bus_idx
#         trip_id = env.active_trips[i]
#         bw_h = env.trips_sars[trip_id][-1][0][3]
#         fw_h = env.trips_sars[trip_id][-1][0][2]
#         if fw_h < bw_h:
#             env.take_action(5)
#         else:
#             env.take_action(1)
#         done = env.prep()
#     if save:
#         env.process_results()
#         post_process.save(path_tr_save, env.trajectories)
#         post_process.save(path_sars_save, env.trips_sars)


# running
run_base(episodes=4, save=True, plot=True)

# outputs
# output.get_results()
# output.get_even_headway_results()

print("ran in %.2f seconds" % (time.time()-st))
