import numpy as np
import matplotlib.pyplot as plt
from post_process import load
import pandas as pd


def plot_learning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0", markersize=8)
    ax.set_xlabel("Training steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.plot(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def plot_cooperative_behavior():
    stops = load('in/xtr/rt_20-2019-09/route_stops.pkl')
    trips = pd.read_csv('out/trajectories2.csv')
    trips_sub = trips[trips['replication'] == 2]
    idx = stops.index('16049')
    focus_stops = [int(stops[i]) for i in range(idx - 5, idx + 5)]
    trips_sub = trips_sub[trips_sub['stop_id'].isin(focus_stops)]
    trips_sub['stop_seq'] = np.nan
    j = 0
    for s in focus_stops:
        trips_sub.loc[trips_sub['stop_id'] == s, 'stop_seq'] = j
        j += 1
    trip_ids = trips_sub['trip_id'].unique().tolist()
    t0 = 27000
    t1 = 33000
    for t in trip_ids:
        trips_sub_id = trips_sub[trips_sub['trip_id'] == t]
        x1, y1 = trips_sub_id['arr_sec'].to_numpy(), trips_sub_id['stop_seq'].to_numpy()
        mask = (x1 > t0) & (x1 < t1)
        x1 = x1[mask]
        y1 = y1[mask]
        x2, y2 = trips_sub_id['dep_sec'].to_numpy(), trips_sub_id['stop_seq'].to_numpy()
        mask = (x2 > t0) & (x2 < t1)
        x2 = x2[mask]
        y2 = y2[mask]
        x = np.append(x1, x2)
        y = np.append(y1, y2)
        sorted_idx = np.argsort(y)
        x = x[sorted_idx]
        y = y[sorted_idx]
        plt.plot(x, y, color='darkturquoise')
    plt.xlabel('seconds', fontsize=8)
    plt.xticks([i for i in range(29500, 32500, 500)], fontsize=8)
    plt.yticks([i for i in range(len(focus_stops))], [i for i in range(idx - 5, idx + 5)], fontsize=8)
    plt.ylabel('stop', fontsize=8)
    plt.show()


def wait_hold_time_relationship():
    x = np.arange(0, 1.2, 1 / 3)
    y0 = np.array([1 / 3, 1, 2, 10 / 3])
    y1 = np.array([2, 1, 1 / 3, 0])
    y_sum = y0 + y1
    idx_min = np.argmin(y_sum)
    fig, ax = plt.subplots()
    ax.plot(x, y0, label='bus $\it{i}$')
    ax.plot(x, y1, label='bus $\it{i+1}$')
    ax.plot(x, y_sum, label='total')
    ax.set_xlabel('hold time of bus $\it{i}$ (units of $\it{H}$)')
    ax.set_ylabel('total wait time')
    ax.set_xticks(x)
    ax.set_xticklabels(['0', '1/3', '2/3', '1'])
    plt.axvline(1 / 3, alpha=0.3, color='gray', linestyle='dashed')
    plt.yticks([])
    plt.legend()
    plt.show()
