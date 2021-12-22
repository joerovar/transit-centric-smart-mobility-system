import random

from simulation_env import DetailedSimulationEnv, DetailedSimulationEnvWithControl, DetailedSimulationEnvWithDeepRL
from file_paths import *
import post_process
from datetime import datetime, timedelta
from output import PostProcessor
st = time.time()


def run_base_detailed(episodes=2, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for i in range(episodes):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save:
        path_trajectories = 'out/NC/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/NC/pax_set_' + tstamp + ext_var
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_completed_pax, pax_set)
    return


def run_base_control_detailed(episodes=2, save=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for i in range(episodes):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save:
        path_trajectories = 'out/EH/trajectories_set_' + tstamp + ext_var
        path_completed_pax = 'out/EH/pax_set_' + tstamp + ext_var
        post_process.save(path_trajectories, trajectories_set)
        post_process.save(path_completed_pax, pax_set)
    return


def run_sample_rl():
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    sars_set = []
    pax_set = []
    for j in range(2):
        env = DetailedSimulationEnvWithDeepRL()
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            if not env.bool_terminal_state:
                action = random.randint(0, 2)
                env.take_action(action)
            env.update_rewards()
            done = env.prep()
        env.process_results()
        trajectories_set.append(env.trajectories)
        sars_set.append(env.trips_sars)
        pax_set.append(env.completed_pax)
        print(env.pool_sars)

    # print(list(trajectories_set[0].keys()))
    return


# run_sample_rl()
# run_base_detailed(episodes=10, save=True)
# run_base_control_detailed(episodes=10, save=True)
path_tr_nc = 'out/NC/trajectories_set_1222-145434.pkl'
path_p_nc = 'out/NC/pax_set_1222-145434.pkl'
path_tr_eh = 'out/EH/trajectories_set_1222-145437.pkl'
path_p_eh = 'out/EH/pax_set_1222-145437.pkl'
path_tr_rl = 'out/RL/trajectory_set_1222-152023.pkl'
path_p_rl = 'out/RL/pax_set_1222-152023.pkl'
path_trips = [path_tr_nc, path_tr_eh, path_tr_rl]
path_pax = [path_p_nc, path_p_eh, path_p_rl]
tags = ['NC', 'EH', 'RL']
#
post_processor = PostProcessor(path_trips, path_pax, tags)
post_processor.total_trip_time_distribution()
post_processor.headway()
post_processor.load_profile()
# post_processor.denied()
# post_processor.hold_time()
# post_processor.wait_times()
# post_processor.rbt_difference('NC', 'RL')
# post_processor.journey_times()

#
# path_trips = [path_tr_nc]
# path_pax = [path_p_nc]
# tags = ['NC']
# post_processor = PostProcessor(path_trips, path_pax, ['NC'])
# post_processor.dwell_time_validation()
# post_processor.trip_time_dist_validation()
# post_processor.load_profile_validation()
print("ran in %.2f seconds" % (time.time()-st))
