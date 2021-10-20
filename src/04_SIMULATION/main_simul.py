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
        bw_h = env.trips_sars[trip_id][-1][0][2]
        fw_h = env.trips_sars[trip_id][-1][0][1]
        if fw_h < bw_h:
            env.take_action(4)
        else:
            env.take_action(1)
        done = env.prep()
    if save:
        env.process_results()
        post_process.save(path_tr_save, env.trajectories)
        post_process.save(path_sars_save, env.trips_sars)


def run_drl_train():
    for i in range(args.n_games):
        score = 0
        env = simulation_env.SimulationEnvDeepRL()
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            i = env.bus_idx
            trip_id = env.active_trips[i]
            all_sars = env.trips_sars[trip_id]
            if len(all_sars) > 1:
                prev_sars = all_sars[-2]
                observation, action, reward, observation_ = prev_sars
                score += reward
                if not args.load_checkpoint:
                    agent.store_transition(observation, action, reward, observation_, int(done))
                    agent.learn()
            observation = all_sars[-1][0]
            action = agent.choose_action(observation)
            env.take_action(action)
            done = env.prep()
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if args.load_checkpoint and n_steps >= 18000:
            break
    plot_learning(steps_array, scores, eps_history, figure_file)
# running
# simulation_run.run_base(save=False)
run_base_drl(save=True)

# outputs
output.get_results()
output.get_rl_results()

print("ran in %.2f seconds" % (time.time()-st))
