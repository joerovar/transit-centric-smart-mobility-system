import matplotlib.pyplot as plt
import numpy as np
import csv

import pandas as pd
from scipy.stats import norm
from input import *


def write_csv_trajectories(trip_data, pathname):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        for trip in trip_data:
            wf.writerow([trip,'{'])
            for stop in trip_data[trip]:
                wf.writerow(stop)
            wf.writerow([' ', '}'])
            wf.writerow('------')
            i += 1
    return


def write_csv_sars(all_sars):
    with open('out/sars_info.dat', 'w', newline='') as f:
        wf = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        for item in all_sars:
            wf.writerow(f'BUS {i}')
            for sub_item in item:
                wf.writerow(sub_item)
            wf.writerow('---------')
            i += 1
    return


def plot_headway_per_stop(hs, pathname):
    for stop in hs:
        for h in hs[stop]:
            plt.scatter(stop, h, color='r', s=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(pathname)

    return


def write_link_times(trip_data, stop_gps, path_writename):
    link_times = {}
    for t in trip_data:
        stop_data = trip_data[t]
        for i in range(len(stop_data) - 1):
            linktime = stop_data[i+1][1] - stop_data[i][1]
            link = stop_data[i][0] + '-' + stop_data[i+1][0]
            if link in link_times:
                link_times[link].append(linktime)
            else:
                link_times[link] = [linktime]
    mean_link_times = {}
    for link in link_times:
        mean_link_times[link] = round(np.array(link_times[link]).mean(), 1)
    link_times_df = pd.DataFrame(mean_link_times.items(), columns=['stop_1', 'time_sec'])
    link_times_df[['stop_1', 'stop_2']] = link_times_df['stop_1'].str.split('-', expand=True)
    link_times_df = link_times_df[['stop_1', 'stop_2', 'time_sec']]
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop_1', 'stop_lat': 'stop_1_lat', 'stop_lon': 'stop_1_lon'})
    link_times_df = pd.merge(link_times_df, s, on='stop_1')
    s = s.rename(columns={'stop_1': 'stop_2', 'stop_1_lat': 'stop_2_lat', 'stop_1_lon': 'stop_2_lon'})
    link_times_df = pd.merge(link_times_df, s, on='stop_2')
    link_times_df.to_csv(path_writename, index=False)
    return


def write_wait_times(headway_data, stop_gps, pathname):
    mean_wait_time = {}
    for stop in headway_data:
        mean_wait_time[stop] = round((np.array(headway_data[stop]).mean()) / 2, 1)
    wait_times = pd.DataFrame(mean_wait_time.items(), columns=['stop', 'wait_time_sec'])
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    wait_times = pd.merge(wait_times, s, on='stop')
    wait_times.to_csv(pathname, index=False)
    return


def get_stop_gps(pathname):
    stop_gps = pd.read_csv(pathname)
    stop_gps = stop_gps[['stop_id', 'stop_lat', 'stop_lon']]
    return stop_gps
# def plot_2hw_cv(hw1, hw2, pathname, labels, colors):
#     fig, ax = plt.subplots(2, sharex='all', figsize=(6, 4))
#     local_hw1, local_hw2 = dict(hw1), dict(hw2)
#     titles = ['inbound', 'outbound']
#     ax[0].grid()
#     ax[1].grid()
#     ax[0].title.set_text(titles[0])
#     ax[1].title.set_text(titles[1])
#     ax[1].set_xticks([0, 1, 2, 3, 4])
#     ax[1].set_xlabel('stop')
#     ax[0].set_ylabel('c.v. of headways')
#     ax[1].set_ylabel('c.v. of headways')
#     k = 0
#     # convert to Coefficient of variance
#     for hs in [local_hw1, local_hw2]:
#         for s in hs:
#             hs[s] = np.array(hs[s])
#             hs[s] = hs[s].std() / hs[s].mean()
#     # create dictionaries where keys are the actual index of stops
#         h_rt1 = {}
#         h_rt2 = {}
#         j = 0
#         for rs in [ROUTE_STOPS[FIXED_ROUTES_BASE[0]+'0'], ROUTE_STOPS[FIXED_ROUTES_BASE[1] + '0']]:
#             for i in range(len(rs)):
#                 if j:
#                     h_rt2[i] = hs[rs[i]]
#                 else:
#                     h_rt1[i] = hs[rs[i]]
#             j += 1
#
#         i = 0
#         for cv in [h_rt1, h_rt2]:
#             lists = sorted(cv.items())
#             x, y = zip(*lists)
#             ax[i].plot(x, y, label=labels[k], color=colors[k], marker='*')
#             i += 1
#         k += 1
#
#     ax[0].legend(loc='best')
#     plt.savefig(pathname)
#     return


def plot_2whiskers(w1, w2, pathname, ylabel, labels, colors):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharey='all')

    bplot = ax[0].boxplot([w1[0], w2[0]], patch_artist=True, showfliers=False, notch=True)
    ax[0].grid(axis='y')
    ax[0].set_xticklabels(labels, rotation=0, fontsize=10)
    ax[0].set_title('inbound', fontsize=10)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax[0].set_ylabel(ylabel)
    bplot2 = ax[1].boxplot([w1[1], w2[1]], patch_artist=True, showfliers=False, notch=True)
    ax[1].grid(axis='y')
    ax[1].set_xticklabels(labels, rotation=0, fontsize=10)
    ax[1].set_title('outbound', fontsize=10)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    plt.savefig(pathname)
    return


def plot_training_info(ep, rew, epsi, pathname):
    fig, ax = plt.subplots()
    p1 = ax.plot(ep, rew, color='g', marker='.', linewidth=1.0, markersize=3.0)
    twin1 = ax.twinx()
    twin1.plot(ep, epsi, color='k', marker='^', linewidth=1.0, markersize=3.0)
    twin1.set_ylabel('% exploration')
    ax.set_ylabel('mean episode reward')
    ax.yaxis.label.set_color('g')
    ax.set_xlabel('training episodes')
    ax.tick_params(axis='y', colors='g')

    ax.grid()
    plt.savefig(pathname)
    return


def plot_policy_extract(policy, pathname):
    extracts = [np.argmax(policy[:, 1, 1], axis=2),
                np.argmax(policy[:, 2, 1], axis=2),
                np.argmax(policy[:, 1, 2], axis=2),
                np.argmax(policy[:, 2, 2], axis=2)
                ]
    fig, axs = plt.subplots(ncols=2, nrows=2, sharex='all', sharey='all')
    shw1 = axs[0, 0].imshow(extracts[0], cmap='Greys')
    axs[0, 0].set_title('requests: (1, 1)', fontsize=11)
    axs[0, 0].set_xticks([0, 1, 2, 3, 4, 5])
    axs[0, 0].set_yticks([0, 1])
    axs[0, 0].set_yticklabels(['in', 'out'])
    axs[0, 0].set_ylabel('direction')
    shw2 = axs[1, 0].imshow(extracts[1], cmap='Greys')
    axs[1, 0].set_title('requests: (2, 1)', fontsize=11)
    axs[1, 0].set_xticks([0, 1, 2, 3, 4, 5])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_yticklabels(['in', 'out'])
    axs[1, 0].set_ylabel('direction')
    axs[1, 0].set_xlabel('delay (min)')
    shw3 = axs[0, 1].imshow(extracts[2], cmap='Greys')
    axs[0, 1].set_title('requests: (1, 2)', fontsize=11)
    axs[0, 1].set_xticks([0, 1, 2, 3, 4, 5])
    axs[0, 1].set_yticks([0, 1])
    axs[0, 1].set_yticklabels(['in', 'out'])

    shw4 = axs[1, 1].imshow(extracts[3], cmap='Greys')
    axs[1, 1].set_title('requests: (2, 2)', fontsize=11)
    axs[1, 1].set_xticks([0, 1, 2, 3, 4, 5])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_yticklabels(['in', 'out'])
    axs[1, 1].set_xlabel('delay (min)')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.subplots_adjust(bottom=0.3)

    c_bar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    c_bar = fig.colorbar(shw1, ticks=[0, 1, 2, 3], orientation='horizontal', cax=c_bar_ax)
    c_bar.ax.set_xticklabels(['No deviation', 'Deviation at 1', 'Deviation at 2', 'Deviation at both'])
    fig.legend()
    plt.savefig(pathname)
    return


def plot_requests(tot_sg, rej_sg, tot_rl, rej_rl, pathname):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='all')
    fig.subplots_adjust(wspace=0.05)
    ax[0].plot(tot_sg, label='total', color='lightgrey')
    ax[0].plot(rej_sg, label='rejected requests', color='lightcoral')
    ax[0].set_title('SG', fontsize=10)
    ax[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    x=range(len(tot_sg))
    ax[0].fill_between(x, 0, rej_sg, alpha=0.5, color='lightcoral')
    ax[0].fill_between(x, rej_sg, tot_sg, alpha=0.5, color='lightgrey')
    ax[1].plot(tot_rl, color='lightgrey')
    ax[1].plot(rej_rl, color='lightcoral')
    ax[1].set_title('PQL', fontsize=10)
    ax[1].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax[1].fill_between(x, 0, rej_rl, alpha=0.5, color='lightcoral')
    ax[1].fill_between(x, rej_rl, tot_rl, alpha=0.5, color='lightgrey')

    ax[0].legend(loc='best')
    plt.savefig(pathname)
    return


def write_results(tot_sg, rej_sg, tot_rl, rej_rl, tot_op, rej_op, runtime,pathname):
    with open(pathname, "w", newline='') as f:
        wf = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wf.writerow(['runtime', runtime])
        wf.writerow(['SG avg req', int(sum(tot_sg)/len(tot_sg))])
        wf.writerow(['SG avg rejected requests', int(sum(rej_sg)/len(rej_sg))])
        wf.writerow(['PQL avg requests', int(sum(tot_rl)/len(tot_rl))])
        wf.writerow(['PQL avg rejected requests', int(sum(rej_rl)/len(rej_rl))])
        wf.writerow(['PQL off peak avg rejected requests', int(sum(tot_op)/len(tot_op))])
        wf.writerow(['PQL off peak avg rejected', int(sum(rej_op)/len(rej_op))])
    return
