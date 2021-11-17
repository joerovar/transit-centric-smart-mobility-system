import output
from simulation_env import SimulationEnv, SimulationEnvWithControl, DetailedSimulationEnv
from file_paths import *
import post_process
from datetime import datetime

st = time.time()


def run_base(episodes=1, save=False, plot=False, time_dep_tt=True, time_dep_dem=True):
    tstamps = []
    for i in range(episodes):
        env = SimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-4])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
    if plot:
        output.get_results(tstamps)
    return


def run_base_control(episodes=1, save=False, plot=False, time_dep_tt=True, time_dep_dem=True):
    tstamps = []
    for i in range(episodes):
        env = SimulationEnvWithControl(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            tstamps.append(datetime.now().strftime('%m%d-%H%M%S%f')[:-4])
            path_trajectories = path_to_outs + dir_var + 'trajectories_' + tstamps[-1] + ext_var
            post_process.save(path_trajectories, env.trajectories)
    if plot:
        output.get_base_control_results(tstamps)
    return


def run_base_detailed(episodes=1, save=False, plot=False, time_dep_tt=True, time_dep_dem=True):
    env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
    done = env.reset_simulation()
    while not done:
        done = env.prep()
        env.process_results()
    # for trip in env.trajectories:
    #     print([trip,env.trajectories[trip][-1]])
    for t in env.trips:
        print(t.trip_id)
        print(len(t.pax))
        for p in t.pax:
            print([p.orig_idx, p.dest_idx, round(p.arr_time)])
    # for s in env.stops:
    #     print(s.stop_id)
    #     print(len(s.pax_completed))
    #     print(len(s.pax))
    #     for p in s.pax_completed:
    #         print([p.orig_idx, p.dest_idx, round(p.arr_time), round(p.board_time), round(p.alight_time), round(p.wait_time), round(p.journey_time)])
    return


run_base_detailed()

print("ran in %.2f seconds" % (time.time()-st))
