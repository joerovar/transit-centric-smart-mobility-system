import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from copy import deepcopy
import seaborn as sns
from datetime import timedelta
from Input_Processor import get_interval, remove_outliers


def denied_count(scenarios, period, scenario_tags, method_tags):
    numer_results = {'parameter': ['denied' for _ in range(len(scenario_tags))],
                     'prc_cancel': scenario_tags}
    for m in method_tags:
        numer_results[m] = []
    for i in range(len(scenario_tags)):
        for j in range(len(method_tags)):
            df = pd.read_pickle('out/' + scenarios[j][i] + '-pax_record_ob.pkl')
            tmp_df = df[(df['arr_time'] <= period[1]) & (df['arr_time'] >= period[0])].copy()
            n_denied = tmp_df[tmp_df['denied'] == 1].shape[0]
            tot = tmp_df.shape[0]
            numer_results[method_tags[j]].append(n_denied/tot * 1000)
    df = pd.DataFrame(numer_results)
    return df


def plot_pax_profile(df, stops, apc=False, path_savefig=None, path_savefig_max=None):
    t0 = 6 * 60 * 60
    t1 = 10 * 60 * 60
    interv_len = 60 * 60
    n_intervals = int((t1 - t0) / interv_len)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8), sharey='all')
    stops_x = np.arange(len(stops))
    w_bar = 0.6
    max_loads = []
    max_ons = []
    max_offs = []
    for n in range(n_intervals):
        tmp_t0 = t0 + n * interv_len
        tmp_t1 = t0 + (n + 1) * interv_len
        if apc:
            tmp_df = df[(df['avl_sec'] >= tmp_t0) & (df['avl_sec'] < tmp_t1)].copy()
        else:
            tmp_df = df[(df['arr_sec'] >= tmp_t0) & (df['arr_sec'] < tmp_t1)].copy()
        avg_loads = []
        avg_ons = []
        avg_offs = []
        for s in stops:
            if apc:
                stop_df = tmp_df[tmp_df['stop_id'] == int(s)].copy()
            else:
                stop_df = tmp_df[tmp_df['stop_id'] == str(s)].copy()
            if not stop_df.empty:
                if apc:
                    avg_loads.append(stop_df['passenger_load'].mean())
                    stop_df['on'] = stop_df['ron'] + stop_df['fon']
                    stop_df['off'] = stop_df['roff'] + stop_df['foff']
                    avg_ons.append(stop_df['on'].mean())
                    avg_offs.append(stop_df['off'].mean())
                else:
                    avg_loads.append(stop_df['pax_load'].mean())
                    avg_ons.append(stop_df['ons'].mean())
                    avg_offs.append(stop_df['offs'].mean())
            else:
                avg_loads.append(np.nan)
                avg_ons.append(np.nan)
                avg_offs.append(np.nan)
        avg_loads[-1] = 0
        avg_ons[-1] = 0
        avg_offs[0] = 0
        if n == 2:
            max_loads = deepcopy(avg_loads)
            max_ons = deepcopy(avg_ons)
            max_offs = deepcopy(avg_offs)
        axs2 = axs.flat[n].twinx()
        axs2.bar(stops_x - w_bar / 2, avg_ons, w_bar)
        axs2.bar(stops_x + w_bar / 2, avg_offs, w_bar)
        axs2.set_ylim(0, 5)
        axs.flat[n].plot(avg_loads, color='black')
        axs.flat[n].set_ylim(0, 35)
        axs.flat[n].set_title(f'{int(tmp_t0 / 60 / 60)} AM')
        axs2.set_ylabel('ons/offs per trip')
        axs.flat[n].set_ylabel('average pax load')
    if path_savefig:
        plt.savefig(path_savefig)
    else:
        plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    ax2.bar(stops_x - w_bar / 2, max_ons, w_bar, color='grey', label='ons')
    ax2.bar(stops_x + w_bar / 2, max_offs, w_bar, color='black', label='offs')
    ax2.set_ylabel('average ons/offs per trip')
    ax.set_ylabel('average pax load')
    ax.plot(max_loads, color='black', label='load')
    ax.set_xlabel('stops')
    ax.set_xticks(np.arange(0, len(stops), 10), labels=np.arange(1, len(stops)+1, 10))
    ax.axvline(10, linestyle='dashed', label='express segment', color='black')
    ax.axvline(0, linestyle='dashed', color='black')
    ax.axvline(28, linestyle='dotted', label='mid holding stops', color='black')
    ax.axvline(44, linestyle='dotted', color='black')
    fig.legend(loc='upper center')
    plt.tight_layout()
    if path_savefig_max:
        plt.savefig(path_savefig_max)
    else:
        plt.show()
    plt.close()
    return


def plot_run_times(scenarios, scenario_tags, method_tags, stops, fig_dir=None):
    numer_results = {'parameter': ['95runt7' for _ in range(len(scenario_tags))],
                     'prc_cancel': scenario_tags}
    for m in method_tags:
        numer_results[m] = []
    intervals = [6, 7, 8]
    fig, axs = plt.subplots(nrows=len(scenario_tags), sharey='all', figsize=(10, 10))
    for ax in axs.flat:
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    for i in range(len(scenario_tags)):
        data = pd.DataFrame(columns=['method', 'run_time', 'dep_t'])
        for j in range(len(method_tags)):
            d = pd.DataFrame(columns=['method', 'run_time', 'dep_t'])
            df = pd.read_pickle('out/' + scenarios[j][i] + '-trip_record_ob.pkl')
            run_ts = trip_t_outbound(df, 6*60*60, 9*60*60, 60, stops, 'arr_sec', 'dep_sec')
            numer_results[method_tags[j]].append(np.percentile(run_ts[1], 95))
            for k in range(len(intervals)):
                method_rows = [method_tags[j]] * len(run_ts[k])
                dep_t_rows = [intervals[k]] * len(run_ts[k])
                tmp_d = pd.DataFrame(list(zip(method_rows, run_ts[k], dep_t_rows)), columns=['method', 'run_time', 'dep_t'])
                d = pd.concat([d, tmp_d], ignore_index=True)
            data = pd.concat([data, d], ignore_index=True)
        sns.boxplot(x='dep_t', y='run_time', hue='method', data=data, ax=axs[i], showfliers=False)
        axs[i].set_title(str(scenario_tags[i]) + '% cancelled')
        axs[i].set_xlabel('departure time (h)')
        axs[i].set_ylabel('run time distribution (min)')
    plt.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir)
    else:
        plt.show()
    plt.close()
    df = pd.DataFrame(numer_results)
    return df


def load_plots(scenarios, scenario_tags, method_tags, stops, period, fig_dir=None, quantile=0.5):
    par_lbl = str(round(quantile*100)) + 'load'
    y_lbl = str(round(quantile*100)) + 'th percentile load'
    numer_results = {'parameter': [par_lbl for _ in range(len(scenario_tags))],
                     'prc_cancel': scenario_tags}
    for m in method_tags:
        numer_results[m] = []
    fig, axs = plt.subplots(ncols=len(scenario_tags), figsize=(13, 8), sharey='all')
    colors = ['black', 'green', 'blue', 'red', 'brown', 'turquoise']
    for i in range(len(method_tags)):
        for j in range(len(scenario_tags)):
            load = []
            df = pd.read_pickle('out/' + scenarios[i][j] + '-trip_record_ob.pkl')
            for s in stops:
                load_tmp = df[(df['stop_id'] == s) & (df['arr_sec'] <= period[1]) &
                              (df['arr_sec'] >= period[0])]['pax_load'].quantile(quantile)
                load.append(load_tmp)
            numer_results[method_tags[i]].append(max(load))
            axs[j].plot(load, label=method_tags[i], color=colors[i])

    for j in range(len(scenario_tags)):
        axs[j].set_ylabel(y_lbl)
        axs[j].set_xlabel('stops')
        axs[j].set_title(f'{scenario_tags[j]} % cancelled')
        axs[j].set_xticks(np.arange(0, len(stops), 10), labels=np.arange(1, len(stops) + 1, 10))
    axs[0].legend()
    plt.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir)
    else:
        plt.show()
    plt.close()
    df = pd.DataFrame(numer_results)
    return df


def trajectory_plots(scenarios, scenario_titles, scheduled_trajectories, period, replication, fig_dir=None):
    fig, axs = plt.subplots(nrows=len(scenarios), sharex='all', figsize=(14, 10))

    df_sched_t_rep = pd.DataFrame(scheduled_trajectories, columns=['trip_id', 'schd_sec', 'dist_traveled'])
    df_sched_t_rep['dist_traveled'] = df_sched_t_rep['dist_traveled'] / 3281
    df_sched_t_rep = df_sched_t_rep[
        (df_sched_t_rep['schd_sec'] >= period[0]) & (df_sched_t_rep['schd_sec'] <= period[1])]
    df_sched_t_rep = df_sched_t_rep.set_index('schd_sec')
    for i in range(len(scenarios)):
        df_out = pd.read_pickle('out/' + scenarios[i] + '-trip_record_ob.pkl')

        df_arr_t = df_out[['trip_id', 'replication', 'dist_traveled', 'arr_sec', 'expressed']].copy()
        df_arr_t = df_arr_t.rename(columns={'arr_sec': 'seconds'})
        df_dep_t = df_out[['trip_id', 'replication', 'dist_traveled', 'dep_sec', 'expressed']].copy()
        df_dep_t = df_dep_t.rename(columns={'dep_sec': 'seconds'})

        df_times = pd.concat([df_arr_t, df_dep_t], axis=0, ignore_index=True)
        df_times['dist_traveled'] = df_times['dist_traveled'] / 3281
        df_times = df_times.sort_values(by=['trip_id', 'seconds'])

        df_times_rep = df_times[df_times['replication'] == replication].copy()

        df_times_rep = df_times_rep[
            (df_times_rep['seconds'] >= period[0]) & (df_times_rep['seconds'] <= period[1])]
        df_times_rep = df_times_rep.set_index('seconds')

        df_sched_t_rep.groupby('trip_id')['dist_traveled'].plot(color='silver', ax=axs[i])
        df_times_rep.groupby('trip_id')['dist_traveled'].plot(color='red', ax=axs[i])

        expressed_trips_df = df_times_rep[df_times_rep['expressed'] == 1].copy()
        expressed_trips_df.groupby('trip_id')['dist_traveled'].plot(color='blue', ax=axs[i])

        non_cancelled_trips = df_times_rep['trip_id'].tolist()
        cancelled_sched_df = df_sched_t_rep[~df_sched_t_rep['trip_id'].isin(non_cancelled_trips)].copy()
        cancelled_sched_df.groupby('trip_id')['dist_traveled'].plot(color='black', ax=axs[i])

        axs[i].set_title(scenario_titles[i])
        axs[i].set_ylabel('km')
    x_ticks = [x for x in range(period[0], period[1] + 30 * 60, 30 * 60)]
    x_labels = [str(timedelta(seconds=round(x)))[:-3] for x in x_ticks]
    plt.xticks(ticks=x_ticks, labels=x_labels)
    if fig_dir:
        plt.savefig(fig_dir)
    else:
        plt.show()
    plt.close()
    return


def cv_hw_plot(scenarios, stops, period, scenario_tags, method_tags, fig_dir=None):
    numer_results = {'parameter': ['cv_h' for _ in range(len(scenario_tags))],
                     'prc_cancel': scenario_tags}
    for m in method_tags:
        numer_results[m] = []
    # linestyles = ['solid', 'dashdot', 'dotted']
    colors = ['black', 'green', 'blue', 'red', 'brown', 'turquoise']
    fig, axs = plt.subplots(ncols=len(scenario_tags), figsize=(13, 8), sharey='all')
    for i in range(len(method_tags)):
        for j in range(len(scenario_tags)):
            df_out = pd.read_pickle('out/' + scenarios[i][j] + '-trip_record_ob.pkl')
            lbl = method_tags[i]
            cv = cv_hw_by_time(df_out, period[0], period[1], stops)
            axs[j].plot(np.arange(len(stops)), cv, label=lbl, color=colors[i])
            numer_results[method_tags[i]].append(np.mean(cv))
    for j in range(len(scenario_tags)):
        axs[j].set_ylabel('c.v. headway')
        axs[j].set_xlabel('stops')
        axs[j].set_title(f'{scenario_tags[j]} % cancelled')
        axs[j].set_xticks(np.arange(0, len(stops), 10), labels=np.arange(1, len(stops) + 1, 10))
    axs[0].legend()

    plt.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir)
    else:
        plt.show()
    plt.close()
    df = pd.DataFrame(numer_results)
    return df


def pax_times_plot(scenarios, boarding_stops, alighting_stops,
                   period, method_tags, scenario_tags, fig_dir=None):
    wt_numer = {'parameter': ['wt' for _ in range(len(scenario_tags))],
                'prc_cancel': scenario_tags}
    rbt_numer = {'parameter': ['rbt' for _ in range(len(scenario_tags))],
                 'prc_cancel': scenario_tags}
    for m in method_tags:
        wt_numer[m], rbt_numer[m] = [], []
    df_plot = pd.DataFrame({'method': [], 'cancelled': [], 'wt': []})
    fig, axs = plt.subplots(ncols=2, figsize=(12, 10))
    axs[0].grid(axis='y')
    axs[0].set_axisbelow(True)
    for i in range(len(method_tags)):
        rbt = []
        for j in range(len(scenario_tags)):
            df_pax = pd.read_pickle('out/' + scenarios[i][j] + '-pax_record_ob.pkl')
            df_p = df_pax[(df_pax['arr_time'] <= period[1]) & (df_pax['arr_time'] >= period[0])].copy()
            df_p = df_p[df_p['orig_idx'].isin(boarding_stops)].copy()
            df_p = df_p[df_p['dest_idx'].isin(alighting_stops)].copy()
            df_p['wt'] = df_p['board_time'] - df_p['arr_time']
            wt_tmp = df_p['wt'].copy() / 60
            wt_numer[method_tags[i]].append(wt_tmp.mean())
            d = {'method': [method_tags[i]] * wt_tmp.shape[0],
                 'cancelled': [int(scenario_tags[j])] * wt_tmp.shape[0],
                 'wt': wt_tmp.tolist()}
            df_plot = pd.concat([df_plot, pd.DataFrame(d)], ignore_index=True)

            rbt_counter = 0
            rbt_sum = 0
            wt_counter = 0
            wt_sum = 0
            for k in range(len(boarding_stops)):
                for n in range(k, len(alighting_stops)):
                    tmp_df = df_pax[(df_pax['orig_idx'] == boarding_stops[k]) &
                                    (df_pax['dest_idx'] == alighting_stops[n])].copy()
                    if tmp_df.shape[0]:
                        tmp_df['jt'] = tmp_df['alight_time'] - tmp_df['arr_time']
                        tmp_df['wt'] = tmp_df['board_time'] - tmp_df['arr_time']
                        rbt_tmp = tmp_df['jt'].quantile(0.95) - tmp_df['jt'].median()

                        rbt_sum += rbt_tmp * tmp_df.shape[0]
                        rbt_counter += tmp_df.shape[0]

                        wt_sum += tmp_df['wt'].mean() * tmp_df.shape[0]
                        wt_counter += tmp_df.shape[0]
            rbt_final = rbt_sum / rbt_counter / 60
            rbt_numer[method_tags[i]].append(rbt_final)
            rbt.append(rbt_final)
            wt_final = wt_sum / wt_counter / 60
            # wt_numer[method_tags[i]].append(wt_final)
        axs[1].plot(scenario_tags, rbt, label=method_tags[i], marker='*')
    axs[1].legend()
    axs[1].set_ylabel('reliability buffer time (min)')
    axs[1].set_xlabel('% cancelled')
    axs[1].set_xticks(scenario_tags)
    sns.boxplot(x='cancelled', y='wt', hue='method', data=df_plot, showfliers=False, ax=axs[0])
    axs[0].set_ylabel('wait time (min)')
    axs[0].set_xlabel('% cancelled')

    if fig_dir:
        plt.savefig(fig_dir)
    else:
        plt.show()
    plt.close()
    wt_df = pd.DataFrame(wt_numer)
    rbt_df = pd.DataFrame(rbt_numer)
    df = pd.concat([wt_df, rbt_df], ignore_index=True)
    return df


def plot_learning(x, scores, filename, lines=None, epsilons=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    if epsilons:
        ax.plot(x, epsilons, color="C0", markersize=8)
        ax.set_ylabel("Epsilon", color="C0")
    ax.set_xlabel("Training steps", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    return


def validate_trip_t_outbound(avl_df, sim_df, start_time, end_time, stops, path_trip_t, path_dwell_t, dates,
                             ignore_terminals=False):
    trip_t_avl = trip_t_outbound(avl_df, start_time, end_time, 60, stops, 'avl_arr_sec',
                                 'avl_dep_sec', is_avl=True, dates=dates,
                                 ignore_terminals=ignore_terminals)
    dwell_t_avl = dwell_t_outbound(avl_df, 2, 66, stops, 'avl_arr_sec', 'avl_dep_sec', 60, start_time, end_time,
                                   is_avl=True, dates=dates)
    trip_t_sim = trip_t_outbound(sim_df, start_time, end_time, 60, stops, 'arr_sec',
                                 'dep_sec', ignore_terminals=ignore_terminals)
    dwell_t_sim = dwell_t_outbound(sim_df, 2, 66, stops, 'arr_sec', 'dep_sec', 60, start_time, end_time)
    plot_calib_hist(trip_t_avl, trip_t_sim, 5, path_trip_t, 'total trip time (seconds)')
    plot_calib_hist(dwell_t_avl, dwell_t_sim, 5, path_dwell_t, 'dwell time (seconds)')
    return


def trip_t_outbound(df_out, start_time, end_time, interval_length, stops_out, col_arr_t, col_dep_t,
                    is_avl=False, dates=None, ignore_terminals=False):
    focus_df_out = df_out[df_out['stop_sequence'] == 1].copy()
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] < end_time]
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] >= start_time]
    if is_avl:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == int(stops_out[0])]
    else:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == stops_out[0]]
    focus_df_out = focus_df_out.sort_values(by='schd_sec')
    focus_trips = focus_df_out['trip_id'].unique().tolist()

    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    trip_t = [[] for _ in range(interval0, interval1)]
    if is_avl:
        nr_days = len(dates)
    else:
        nr_days = df_out['replication'].max()
    for i in range(nr_days):
        if is_avl:
            day_df = df_out[df_out['avl_arr_time'].astype(str).str[:10] == dates[i]].copy()
        else:
            day_df = df_out[df_out['replication'] == i + 1].copy()
        for trip in focus_trips:
            trip_df = day_df[day_df['trip_id'] == trip].copy()
            if ignore_terminals:
                t0 = trip_df[trip_df['stop_sequence'] == 2]
                t1 = trip_df[trip_df['stop_sequence'] == 66]
            else:
                t0 = trip_df[trip_df['stop_sequence'] == 1]
                t1 = trip_df[trip_df['stop_sequence'] == 67]
            if not t0.empty and not t1.empty:
                t0 = t0.iloc[0]
                t1 = t1.iloc[0]
                interval = get_interval(t0['schd_sec'], interval_length)
                dep_t = t0[col_dep_t].astype(int)
                arr_t = t1[col_arr_t].astype(int)
                trip_t[interval - interval0].append((arr_t - dep_t)/60)
    if is_avl:
        for i in range(interval1 - interval0):
            if trip_t[i]:
                trip_t[i] = remove_outliers(np.array(trip_t[i])).tolist()
    return trip_t


def dwell_t_outbound(df_out, start_stop, end_stop, stops, col_arr_t, col_dep_t, interval_length,
                     start_time, end_time, is_avl=False, dates=None):
    focus_df_out = df_out[df_out['stop_sequence'] == 1].copy()
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] < end_time]
    focus_df_out = focus_df_out[focus_df_out['schd_sec'] >= start_time]
    if is_avl:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == int(stops[0])]
    else:
        focus_df_out = focus_df_out[focus_df_out['stop_id'] == stops[0]]
    focus_df_out = focus_df_out.sort_values(by='schd_sec')
    focus_trips = focus_df_out['trip_id'].unique().tolist()
    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    dwell_t = [[] for _ in range(interval0, interval1)]
    if is_avl:
        nr_days = len(dates)
    else:
        nr_days = df_out['replication'].max()
    for i in range(nr_days):
        if is_avl:
            day_df = df_out[df_out['avl_arr_time'].astype(str).str[:10] == dates[i]].copy()
        else:
            day_df = df_out[df_out['replication'] == i + 1].copy()
        for trip in focus_trips:
            trip_df = day_df[day_df['trip_id'] == trip].copy()
            t0 = trip_df[trip_df['stop_sequence'] == start_stop]
            if not t0.empty:
                interval = get_interval(t0['schd_sec'].iloc[0], interval_length)
                mid_route_df = trip_df[
                    (trip_df['stop_sequence'] >= start_stop) & (trip_df['stop_sequence'] <= end_stop)].copy()
                if is_avl:
                    mid_route_df = mid_route_df.drop_duplicates(subset='stop_sequence', keep='first')
                if mid_route_df.shape[0] == end_stop - start_stop + 1:
                    mid_route_df.loc[:, 'dwell_t'] = mid_route_df[col_dep_t] - mid_route_df[col_arr_t]
                    dwell_t[interval - interval0].append(mid_route_df['dwell_t'].sum())
    return dwell_t


def validate_delay_inbound(avl_df, sim_df, start_t_sec, end_t_sec, start_interval, interval_mins=60):
    # start_interval = 5
    arr_delays_long, arr_delays_short, dep_delays_long, dep_delays_short = delay_inbound(avl_df, start_t_sec,
                                                                                         end_t_sec,
                                                                                         interval_mins,
                                                                                         'avl_arr_sec',
                                                                                         'avl_dep_sec',
                                                                                         [('17164', 2),
                                                                                          ('14800', 22)],
                                                                                         [('6360', 2),
                                                                                          ('14800', 62)],
                                                                                         outlier_removal=True)

    arr_del_long_sim, arr_del_short_sim, dep_del_long_sim, dep_del_short_sim = delay_inbound(sim_df,
                                                                                             start_t_sec,
                                                                                             end_t_sec,
                                                                                             interval_mins,
                                                                                             'arr_sec',
                                                                                             'arr_sec',
                                                                                             [('15136', 1),
                                                                                              ('386', 23)],
                                                                                             [('8613', 1),
                                                                                              ('386', 63)])

    plot_calib_hist(arr_delays_long, arr_del_long_sim, start_interval, 'out/compare/validate/arr_delays_in_long.png',
                    'arr delay (seconds)')
    plot_calib_hist(arr_delays_short, arr_del_short_sim, start_interval, 'out/compare/validate/arr_delays_in_short.png',
                    'arr delay (seconds)')
    plot_calib_hist(dep_delays_long, dep_del_long_sim, start_interval, 'out/compare/validate/dep_delays_in_long.png',
                    'dep delay (seconds)')
    plot_calib_hist(dep_delays_short, dep_del_short_sim, start_interval, 'out/compare/validate/dep_delays_in_short.png',
                    'dep delay (seconds)')
    return


def validate_delay_outbound(avl_df, sim_df, start_t_sec, end_t_sec, interval_mins=60):
    start_interval = 5
    arr_delays_out, dep_delays_out = delay_outbound(avl_df, start_t_sec, end_t_sec, interval_mins,
                                                    'avl_arr_sec', 'avl_dep_sec', [('388', 2), ('3954', 66)],
                                                    outlier_removal=True)
    arr_delays_out_sim, dep_delays_out_sim = delay_outbound(sim_df, start_t_sec, end_t_sec,
                                                            interval_mins, 'arr_sec', 'dep_sec',
                                                            [('386', 1), ('8613', 67)])
    plot_calib_hist(arr_delays_out, arr_delays_out_sim, start_interval, 'out/compare/validate/arr_delays_out.png',
                    'arr delay (seconds)')
    plot_calib_hist(dep_delays_out, dep_delays_out_sim, start_interval, 'out/compare/validate/dep_delays_out.png',
                    'dep delay (seconds)')
    return


def plot_calib_hist(delay_avl, delay_sim, start_interval, filename, xlabel):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    for i in range(ax.size):
        ax.flat[i].hist([delay_avl[i], delay_sim[i]], density=True, label=['avl', 'sim'])
        ax.flat[i].set_title(f'hour {start_interval + i}')
        ax.flat[i].set_yticks([])
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()
    return


def validate_cv_hw_outbound(avl_df, sim_df, start_t_sec, end_t_sec, interval_min, stops, dates, start_interval=5):
    hw_out_cv = cv_hw_from_avl(avl_df, start_t_sec, end_t_sec, interval_min, stops, dates)
    hw_out_cv_sim = cv_hw_by_intervals(sim_df, start_t_sec, end_t_sec, interval_min, stops)
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey='all', sharex='all')
    for i in range(ax.size):
        ax.flat[i].plot(hw_out_cv[i], label='avl')
        ax.flat[i].plot(hw_out_cv_sim[i], label='sim')
        ax.flat[i].set_title(f'hour {start_interval + i}')
        ax.flat[i].set_xlabel('stop')
    ax[0, 0].set_ylabel('c.v. headway')
    ax[1, 0].set_ylabel('c.v. headway')
    plt.legend()
    plt.tight_layout()
    plt.savefig('out/compare/validate/cv_hw.png')
    plt.close()
    return


def delay_inbound(trips_df, start_time, end_time, delay_interval_length, col_arr_t, col_dep_t, terminals_short,
                  terminals_long, outlier_removal=False):
    trips_df2 = trips_df.copy()
    trips_df2['stop_id'] = trips_df2['stop_id'].astype(str)

    end_terminal_id = terminals_short[1][0]
    terminal_seq_long = terminals_long[1][1]
    terminal_seq_short = terminals_short[1][1]

    arr_delays_long = []
    arr_delays_short = []
    # arrivals
    arr_long_df = trips_df2[trips_df2['stop_id'] == end_terminal_id].copy()
    arr_long_df = arr_long_df[arr_long_df['stop_sequence'] == terminal_seq_long]
    arr_long_df[col_arr_t] = arr_long_df[col_arr_t] % 86400

    arr_short_df = trips_df2[trips_df2['stop_id'] == end_terminal_id].copy()
    arr_short_df = arr_short_df[arr_short_df['stop_sequence'] == terminal_seq_short]
    arr_short_df[col_arr_t] = arr_short_df[col_arr_t] % 86400

    start_terminal_id_long = terminals_long[0][0]
    start_terminal_id_short = terminals_short[0][0]
    start_terminal_seq_long = terminals_long[0][1]
    start_terminal_seq_short = terminals_short[0][1]

    dep_delays_long = []
    dep_delays_short = []
    # departures
    dep_long_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_long].copy()
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == start_terminal_seq_long]
    dep_long_df[col_dep_t] = dep_long_df[col_dep_t] % 86400

    dep_short_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_short].copy()
    dep_short_df = dep_short_df[dep_short_df['stop_sequence'] == start_terminal_seq_short]
    dep_short_df[col_dep_t] = dep_short_df[col_dep_t] % 86400

    interval0 = get_interval(start_time, delay_interval_length)
    interval1 = get_interval(end_time, delay_interval_length)

    for interval in range(interval0, interval1):
        # arrivals
        temp_df = arr_long_df[arr_long_df['schd_sec'] >= interval * delay_interval_length * 60].copy()
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays_long.append(d.tolist())

        temp_df = arr_short_df[arr_short_df['schd_sec'] >= interval * delay_interval_length * 60].copy()
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays_short.append(d.tolist())

        # departures
        temp_df = dep_long_df[dep_long_df['schd_sec'] >= interval * delay_interval_length * 60].copy()
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays_long.append(d.tolist())

        temp_df = dep_short_df[dep_short_df['schd_sec'] >= interval * delay_interval_length * 60].copy()
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays_short.append(d.tolist())
    return arr_delays_long, arr_delays_short, dep_delays_long, dep_delays_short


def delay_outbound(trips_df, start_time, end_time, delay_interval_length, col_arr_t, col_dep_t, terminals_info,
                   outlier_removal=False):
    trips_df2 = trips_df.copy()
    trips_df2['stop_id'] = trips_df2['stop_id'].astype(str)

    start_terminal_seq = terminals_info[0][1]
    end_terminal_id = terminals_info[1][0]
    end_terminal_seq = terminals_info[1][1]

    arr_delays = []
    # arrivals
    arr_long_df = trips_df2[trips_df2['stop_id'] == end_terminal_id]
    arr_long_df = arr_long_df[arr_long_df['stop_sequence'] == end_terminal_seq]
    arr_long_df[col_arr_t] = arr_long_df[col_arr_t] % 86400
    start_terminal_id_long = terminals_info[0][0]

    dep_delays = []
    # departures
    dep_long_df = trips_df2[trips_df2['stop_id'] == start_terminal_id_long]
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == start_terminal_seq]
    dep_long_df = dep_long_df[dep_long_df['stop_sequence'] == start_terminal_seq]
    dep_long_df[col_dep_t] = dep_long_df[col_dep_t] % 86400

    interval0 = get_interval(start_time, delay_interval_length)
    interval1 = get_interval(end_time, delay_interval_length)

    for interval in range(interval0, interval1):
        # arrivals
        temp_df = arr_long_df[arr_long_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_arr_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        arr_delays.append(d.tolist())

        # departures
        temp_df = dep_long_df[dep_long_df['schd_sec'] >= interval * delay_interval_length * 60]
        temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * delay_interval_length * 60]
        temp_df['delay'] = temp_df[col_dep_t] - temp_df['schd_sec']
        if outlier_removal:
            d = remove_outliers(temp_df['delay'].to_numpy())
        else:
            d = temp_df['delay']
        dep_delays.append(d.tolist())
    return arr_delays, dep_delays


def cv_hw_by_time(trip_record_df, start_time, end_time, stops, prnt_idx=None):
    nr_replications = trip_record_df['replication'].max()
    hws = [[] for _ in range(len(stops))]
    for rep_nr in range(1, nr_replications + 1):
        date_df = trip_record_df[trip_record_df['replication'] == rep_nr].copy()
        temp_df = date_df[(date_df['arr_sec'] >= start_time) & (date_df['arr_sec'] <= end_time)].copy()
        for j in range(len(stops)):
            df = temp_df[temp_df['stop_id'] == stops[j]].copy()
            df = df[df['stop_sequence'] == j + 1]
            df = df.sort_values(by='arr_sec')
            arr_sec = df['arr_sec'].tolist()
            if len(arr_sec) > 1:
                for i in range(1, len(arr_sec)):
                    hws[j].append(arr_sec[i] - arr_sec[i - 1])
    cv_hws = []
    for stop_idx in range(len(hws)):
        sd = np.std(hws[stop_idx])
        mean = np.mean(hws[stop_idx])
        if prnt_idx:
            if stop_idx in prnt_idx:
                print('----')
                print(f'for stop idx we have sd {round(sd, 2)} and mean {round(mean, 2)}')
                print(f'headways {[round(h, 1) for h in hws[stop_idx]]}')
        cv_hws.append(sd / mean)
    return cv_hws


def cv_hw_by_intervals(trip_record_df, start_time, end_time, interval_length, stops):
    nr_replications = trip_record_df['replication'].max()
    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    hws = [[[] for _ in range(len(stops))] for _ in range(interval0, interval1)]
    for rep_nr in range(1, nr_replications + 1):
        date_df = trip_record_df[trip_record_df['replication'] == rep_nr]
        for interval in range(interval0, interval1):
            temp_df = date_df[date_df['schd_sec'] >= interval * interval_length * 60]
            temp_df = temp_df[temp_df['schd_sec'] < (interval + 1) * interval_length * 60]
            for j in range(len(stops)):
                df = temp_df[temp_df['stop_id'] == stops[j]]
                df = df[df['stop_sequence'] == j + 1]
                df = df.sort_values(by='arr_sec')
                arr_sec = df['arr_sec'].tolist()
                if len(arr_sec) > 1:
                    for i in range(1, len(arr_sec)):
                        hws[interval - interval0][j].append(arr_sec[i] - arr_sec[i - 1])
    cv_hws = []
    for interval in range(len(hws)):
        cv_hws.append([])
        for stop_idx in range(len(hws[interval])):
            cv_hws[-1].append(np.std(hws[interval][stop_idx]) / np.mean(hws[interval][stop_idx]))
    return cv_hws


def cv_hw_from_avl(avl_df, start_time, end_time, interval_length, stops, dates):
    avl_df2 = avl_df.copy()
    avl_df2['stop_id'] = avl_df2['stop_id'].astype(str)
    interval0 = get_interval(start_time, interval_length)
    interval1 = get_interval(end_time, interval_length)
    hws = [[[] for _ in range(len(stops))] for _ in range(interval0, interval1)]
    for d in dates:
        date_df = avl_df2[avl_df2['avl_arr_time'].astype(str).str[:10] == d]
        for interval in range(interval0, interval1):
            temp_df = date_df[date_df['schd_sec'] >= interval * interval_length * 60]
            temp_df = temp_df[temp_df['schd_sec'] <= (interval + 1) * interval_length * 60]
            for j in range(len(stops)):
                df = temp_df[temp_df['stop_id'] == stops[j]]
                df = df[df['stop_sequence'] == j + 1]
                df = df.sort_values(by='avl_arr_sec')
                arr_sec = df['avl_arr_sec'].tolist()
                if len(arr_sec) > 1:
                    for i in range(1, len(arr_sec)):
                        hws[interval - interval0][j].append(arr_sec[i] - arr_sec[i - 1])
    cv_hws = []
    for interval in range(len(hws)):
        cv_hws.append([])
        for stop_idx in range(len(hws[interval])):
            hws_arr = remove_outliers(np.array(hws[interval][stop_idx]))
            cv_hws[-1].append(np.std(hws_arr) / np.mean(hws_arr))
    return cv_hws


def save(pathname, par):
    with open(pathname, 'wb') as tf:
        pickle.dump(par, tf)
    return


def load(pathname):
    with open(pathname, 'rb') as tf:
        var = pickle.load(tf)
    return var


def write_trajectory_set(trajectory_set, pathname, idx_arr_t, idx_dep_t, idx_hold, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trajectories in trajectory_set:
            day = i
            for trip in trajectories:
                for stop_info in trajectories[trip]:
                    stop_lst = deepcopy(stop_info)
                    stop_lst[idx_arr_t] = str(timedelta(seconds=round(stop_lst[idx_arr_t])))
                    stop_lst[idx_dep_t] = str(timedelta(seconds=round(stop_lst[idx_dep_t])))
                    stop_lst[idx_hold] = round(stop_lst[idx_hold])
                    stop_lst.insert(0, trip)
                    stop_lst.append(day)
                    stop_lst.append(round(stop_info[idx_arr_t]))
                    stop_lst.append(round(stop_info[idx_dep_t]))
                    stop_lst.append(round(stop_info[idx_dep_t] - stop_info[idx_arr_t]))
                    wf.writerow(stop_lst)
            i += 1
    return


def write_trajectory_set_inbound(trajectory_set, pathname, idx_arr_t, idx_schd_sec, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trajectories in trajectory_set:
            day = i
            for trip in trajectories:
                for stop_info in trajectories[trip]:
                    stop_lst = deepcopy(stop_info)
                    stop_lst[idx_arr_t] = str(timedelta(seconds=round(stop_lst[idx_arr_t])))
                    stop_lst[idx_schd_sec] = str(timedelta(seconds=round(stop_lst[idx_schd_sec])))
                    stop_lst.insert(0, trip)
                    stop_lst.append(day)
                    stop_lst.append(round(stop_info[idx_arr_t]))
                    stop_lst.append(round(stop_info[idx_schd_sec]))
                    wf.writerow(stop_lst)
            i += 1
    return


def write_sars(trip_data, pathname, header=None):
    with open(pathname, 'w', newline='') as f:
        wf = csv.writer(f, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        i = 1
        if header:
            wf.writerow(header)
        for trip in trip_data:
            for s in trip_data[trip]:
                stop_lst = deepcopy(s)
                stop_lst.insert(0, trip)
                wf.writerow(stop_lst)
            i += 1
    return


def load_from_trajectory_set(trajectory_set, stops, idx_load, peak_load_stop):
    load_per_stop = {s: [] for s in stops}
    peak_loads = []
    for trajectories in trajectory_set:
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                bus_load = stop_details[idx_load]
                load_per_stop[stop_id].append(bus_load)
                if peak_load_stop == stop_id:
                    peak_loads.append(bus_load)
    load_avg_per_stop = [np.mean(load_per_stop[s]) for s in stops]
    load_sd_per_stop = [np.std(load_per_stop[s]) for s in stops]
    max_load_per_stop = [np.percentile(load_per_stop[s], 95) for s in stops]
    min_load_per_stop = [np.percentile(load_per_stop[s], 10) for s in stops]
    return load_avg_per_stop, load_sd_per_stop, peak_loads, max_load_per_stop, min_load_per_stop


def trip_time_from_trajectory_set(trajectory_set, idx_dep_t, idx_arr_t):
    trip_times = []
    for trajectories in trajectory_set:
        for trip in trajectories:
            dep_time = trajectories[trip][1][idx_dep_t]
            arr_time = trajectories[trip][-2][idx_arr_t]
            trip_times.append(arr_time - dep_time)
    return trip_times


def plot_4_trip_t_dist(all_trip_t, tags, path_save):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    i = 0
    for temp_trip_t in all_trip_t:
        sns.histplot([t / 60 for t in temp_trip_t], kde=True, color='gray', alpha=0.5, ax=axs.flat[i])
        axs.flat[i].axvline(np.percentile(temp_trip_t, 95) / 60, color='black', linestyle='dashed', alpha=0.7)
        axs.flat[i].set_title(tags[i], fontsize=9)
        if i > 1:
            axs.flat[i].set_xlabel('total trip run time (min)', fontsize=8)
        i += 1
    plt.xlim(60, 81)
    for ax in axs.flat:
        ax.tick_params(labelsize=8)
        ax.set_ylabel('frequency', fontsize=8)
    # plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path_save + 'trip_t_dist.png')
    plt.close()
    return


def plot_5_trip_t_dist(all_trip_t, tags, path_save):
    fig, axs = plt.subplots(nrows=5, sharex='all', sharey='all')
    i = 0
    for temp_trip_t in all_trip_t:
        sns.histplot([t / 60 for t in temp_trip_t], kde=True, color='gray', alpha=0.5, ax=axs.flat[i])
        axs.flat[i].axvline(np.percentile(temp_trip_t, 95) / 60, color='black', linestyle='dashed', alpha=0.7)
        axs.flat[i].set_title(tags[i], fontsize=9)
        if i > 1:
            axs.flat[i].set_xlabel('total trip time (seconds)', fontsize=8)
        i += 1
    plt.xlim(60, 81)
    for ax in axs.flat:
        ax.tick_params(labelsize=8)
        ax.set_ylabel('frequency', fontsize=8)
    # plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path_save + 'trip_t_dist.png')
    plt.close()
    return


def control_from_trajectory_set(df_path, controlled_stops):
    df = pd.read_csv(df_path)
    df['stop_id'] = df['stop_id'].astype(str)
    df_control = df[df['stop_id'].isin(controlled_stops[:-1])]
    # print(df_control)
    hold_times = df_control['hold_time'].tolist()
    # nr_trips = len(df['trip_id'].unique().tolist()) * df['replication'].max()
    skipped_count = df_control['skipped'].sum()
    skipped_freq = skipped_count / df_control.shape[0] * 100
    return hold_times, skipped_freq


def pax_per_trip_from_trajectory_set(trajectory_set, idx_load, idx_ons, idx_offs, stops):
    avg_lp = {s: [] for s in stops}
    avg_lp_sd = {s: [] for s in stops}
    avg_ons = {s: [] for s in stops}
    avg_offs = {s: [] for s in stops}
    avg_lp_per_stop = []
    avg_lp_sd_per_stop = []
    avg_ons_per_stop = []
    avg_offs_per_stop = []
    tot_ons_per_replication = []
    for trajectories in trajectory_set:
        bus_load_all = {s: [] for s in stops}
        ons_all = {s: [] for s in stops}
        offs_all = {s: [] for s in stops}
        total_ons = 0
        for trip in trajectories:
            for stop_details in trajectories[trip]:
                stop_id = stop_details[0]
                bus_load = stop_details[idx_load]
                ons = stop_details[idx_ons]
                total_ons += ons
                offs = stop_details[idx_offs]
                bus_load_all[stop_id].append(bus_load)
                ons_all[stop_id].append(ons)
                offs_all[stop_id].append(offs)
        tot_ons_per_replication.append(total_ons)
        for stop in bus_load_all:
            avg_lp[stop].append(np.mean(bus_load_all[stop]))
            avg_lp_sd[stop].append(np.std(bus_load_all[stop]))
            avg_ons[stop].append(np.mean(ons_all[stop]))
            avg_offs[stop].append(np.mean(offs_all[stop]))
    avg_tot_ons = np.mean(tot_ons_per_replication)
    for s in stops:
        avg_lp_per_stop.append(np.mean(avg_lp[s]))
        avg_lp_sd_per_stop.append(np.mean(avg_lp_sd[s]))
        avg_ons_per_stop.append(np.mean(avg_ons[s]))
        avg_offs_per_stop.append(np.mean(avg_offs[s]))
    return avg_lp_per_stop, avg_lp_sd_per_stop, avg_tot_ons, avg_ons_per_stop, avg_offs_per_stop


def write_link_times(link_times_mean, link_times_std, stop_gps, pathname, ordered_stops):
    link_times_df = pd.DataFrame(link_times_mean.items(), columns=['stop_1', 'mean_sec'])
    link_times_df['std_sec'] = link_times_df['stop_1'].map(link_times_std)
    link_times_df[['stop_1', 'stop_2']] = link_times_df['stop_1'].str.split('-', expand=True)
    link_times_df = link_times_df[['stop_1', 'stop_2', 'mean_sec', 'std_sec']]

    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)

    s1 = s.rename(columns={'stop_id': 'stop_1', 'stop_lat': 'stop_1_lat', 'stop_lon': 'stop_1_lon'})
    link_times_df = pd.merge(link_times_df, s1, on='stop_1')
    s2 = s1.rename(columns={'stop_1': 'stop_2', 'stop_1_lat': 'stop_2_lat', 'stop_1_lon': 'stop_2_lon'})
    link_times_df = pd.merge(link_times_df, s2, on='stop_2')

    stop_seq = [i for i in range(len(ordered_stops) - 1)]
    ordered_stop_data = {'stop_1': ordered_stops[:-1], 'stop_1_sequence': stop_seq}
    os_df = pd.DataFrame(ordered_stop_data)
    os_df['stop_1'] = os_df['stop_1'].astype(str)
    link_times_df = pd.merge(link_times_df, os_df, on='stop_1')
    link_times_df = link_times_df.sort_values(by=['stop_1_sequence'])

    link_times_df.to_csv(pathname, index=False)
    return


def write_dwell_times(dwell_times_mean, dwell_times_std, stop_gps, pathname, ordered_stops):
    dwell_times_std_df = pd.DataFrame(dwell_times_std.items(), columns=['stop', 'std_sec'])
    dwell_times_mean_df = pd.DataFrame(dwell_times_mean.items(), columns=['stop', 'mean_sec'])
    dwell_times_df = pd.merge(dwell_times_mean_df, dwell_times_std_df, on='stop')
    s = stop_gps.copy()
    s['stop_id'] = s['stop_id'].astype(str)
    s = s.rename(columns={'stop_id': 'stop'})
    dwell_times_df = pd.merge(dwell_times_df, s, on='stop')

    stop_seq = [i for i in range(len(ordered_stops))]
    ordered_stop_data = {'stop': ordered_stops, 'stop_sequence': stop_seq}
    os_df = pd.DataFrame(ordered_stop_data)
    os_df['stop'] = os_df['stop'].astype(str)

    dwell_times_df = pd.merge(dwell_times_df, os_df, on='stop')
    dwell_times_df = dwell_times_df.sort_values(by=['stop_sequence'])
    dwell_times_df.to_csv(pathname, index=False)
    return


def get_stop_loc(pathname):
    stop_gps = pd.read_csv(pathname)
    stop_gps = stop_gps[['stop_id', 'stop_lat', 'stop_lon']]
    return stop_gps


def get_pax_times_fast(pax_set, n_stops, include_rbt=False):
    fields = ['orig_idx', 'dest_idx', 'journey_time', 'wait_time', 'denied']
    wait_time_set = []
    denied_rate_per_rep = []
    denied_wait_time_set = []
    rbt_od_set = []
    pax_wt_0_2 = 0
    pax_wt_2_4 = 0
    pax_wt_4_inf = 0
    for rep in pax_set:
        df = pd.DataFrame([{f: getattr(p, f) for f in fields} for p in rep])
        wait_time_set.append(df['wait_time'].mean() / 60)
        pax_wt_0_2 += df[df['wait_time'] < 2.5 * 60].shape[0]
        pax_wt_2_4 += df[(df['wait_time'] >= 2.5 * 60) & (df['wait_time'] < 5 * 60)].shape[0]
        pax_wt_4_inf += df[df['wait_time'] > 5 * 60].shape[0]
        tot_pax = df.shape[0]
        denied_df = df[df['denied'] == 1]
        denied_pax = denied_df.shape[0]
        denied_wt = denied_df['wait_time'].mean()
        denied_rate_per_rep.append(denied_pax / tot_pax)
        denied_wait_time_set.append(denied_wt)

        if include_rbt:
            rbt_od = np.zeros(shape=(n_stops, n_stops))
            rbt_od[:] = np.nan
            pax_count = 0
            for s0 in range(n_stops - 1):
                o_jt = df[df['orig_idx'] == s0]
                for s1 in range(s0 + 1, n_stops):
                    od_jt = o_jt[o_jt['dest_idx'] == s1]
                    if not od_jt.empty:
                        pax_od = od_jt.shape[0]
                        pax_count += pax_od
                        rbt_od[s0, s1] = (od_jt['journey_time'].quantile(0.95) - od_jt[
                            'journey_time'].median()) * pax_od
            rbt_od_set.append(np.nansum(rbt_od) / pax_count / 60)
    denied_rate = np.mean(denied_rate_per_rep)
    denied_wait_time_mean = np.nanmean(denied_wait_time_set)
    tot_pax_reps = pax_wt_0_2 + pax_wt_2_4 + pax_wt_4_inf
    pc_pax_wt_0_2 = round(pax_wt_0_2 / tot_pax_reps * 100, 2)
    pc_pax_wt_2_4 = round(pax_wt_2_4 / tot_pax_reps * 100, 2)
    pc_pax_wt_4_inf = round(pax_wt_4_inf / tot_pax_reps * 100, 2)
    return wait_time_set, denied_rate, denied_wait_time_mean, rbt_od_set, pc_pax_wt_0_2, pc_pax_wt_2_4, pc_pax_wt_4_inf


def plot_headway(cv_hw_set, ordered_stops, lbls, colors, pathname=None, controlled_stops=None, cv_scale=(0, 1, 0.1)):
    fig, ax1 = plt.subplots()
    x = np.arange(len(ordered_stops))
    j = 0
    for cv in cv_hw_set:
        ax1.plot(x, cv, color=colors[j], label=lbls[j], marker='*')
        j += 1

    ax1.set_xlabel('stop', fontsize=8)
    ax1.set_ylabel('coefficient of variation of headway', fontsize=8)
    ax1.set_yticks(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]))
    ax1.set_yticklabels(np.arange(cv_scale[0], cv_scale[1] + cv_scale[2], cv_scale[2]).round(decimals=1), fontsize=8)
    if controlled_stops:
        for cs in controlled_stops:
            idx = ordered_stops.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    x_ticks = np.arange(0, len(ordered_stops), 3)
    x_tick_labels = x_ticks + 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, fontsize=8)
    plt.legend()
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_load_profile_grid(lp_set, lp_max_set, lp_min_set, os, tags, pathname=None):
    x1 = np.arange(len(os))
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    obj = []
    i = 0
    for ax in axs.flat:
        obj1, = ax.plot(x1, lp_set[i], color='black')
        obj2, = ax.plot(x1, lp_max_set[i], color='red')
        obj3, = ax.plot(x1, lp_min_set[i], color='green')
        obj.append([obj1, obj2, obj3])
        ax.set_title(tags[i], fontsize=9)
        ax.grid(axis='y')
        ax.axhline(y=50, color='red', alpha=0.5)
        ax.set_ylabel('load (pax)', fontsize=9)
        ax.set_xlabel('stop', fontsize=9)
        ax.tick_params(labelsize=9)
        i += 1
    fig.legend(obj[-1], ['median', '95-th', '10-th'], bbox_to_anchor=(0.535, 0.0), loc='lower center', fontsize=9,
               ncol=3,
               columnspacing=0.8)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_load_profile_benchmark(load_set, os, lbls, colors, load_sd_set=None, pathname=None, x_y_lbls=None,
                                controlled_stops=None):
    x1 = np.arange(len(os))
    fig, ax1 = plt.subplots()
    for j in range(len(load_set)):
        ax1.plot(x1, load_set[j], label=lbls[j], color=colors[j])
    if controlled_stops:
        for cs in controlled_stops:
            idx = os.index(cs)
            plt.axvline(x=idx, color='gray', alpha=0.5, linestyle='dashed')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(os, fontsize=6, rotation=90)

    # right, left, top, bottom
    if x_y_lbls:
        ax1.set_xlabel(x_y_lbls[0])
        ax1.set_ylabel(x_y_lbls[1])
    fig.legend(loc='upper center')
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_sensitivity_whisker(dset1, dset2, method_labels, scenario_labels, y_label1, y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    nr_methods = len(method_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='all')
    fig.subplots_adjust(wspace=0.02)
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[nr_methods * i:nr_methods * (i + 1)], sym='')
        axes[0, i].set_xticks(np.arange(1, nr_methods + 1))
        axes[0, i].set_xticklabels(method_labels, fontsize=8)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set(ylabel=y_label1)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[nr_methods * i:nr_methods * (i + 1)], sym='')
        axes[1, i].set_xticks(np.arange(1, nr_methods + 1))
        axes[1, i].set_xticklabels(method_labels, fontsize=8)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set(ylabel=y_label2)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_sensitivity_whisker_run_t(dset1, dset2, method_labels, scenario_labels, base_method_labels,
                                   y_label1, y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    nr_methods = len(method_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='col')
    fig.subplots_adjust(wspace=0.02)
    ranges = [(0, 5), (5, 8), (8, 13)]
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[ranges[i][0]:ranges[i][1]], sym='')
        axes[0, i].set_xticks(np.arange(1, ranges[i][1] - ranges[i][0] + 1))
        # axes[0, i].set_xticklabels(method_labels, fontsize=8)
        if i == 1:
            axes[0, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[0, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set_ylabel(y_label1, fontsize=8)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[ranges[i][0]:ranges[i][1]], sym='')
        axes[1, i].set_xticks(np.arange(1, ranges[i][1] - ranges[i][0] + 1))
        if i == 1:
            axes[1, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[1, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        # axes[1, i].set_xticklabels(method_labels, fontsize=8)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set_ylabel(y_label2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_sensitivity_whisker_compliance(dset1, dset2, method_labels, scenario_labels, base_method_labels, y_label1,
                                        y_label2, path_save):
    nr_scenarios = len(scenario_labels)
    fig, axes = plt.subplots(ncols=nr_scenarios, nrows=2, sharey='row', sharex='col')
    fig.subplots_adjust(wspace=0.02)
    ranges = [(0, 3), (3, 8), (8, 13)]
    for i in range(nr_scenarios):
        axes[0, i].boxplot(dset1[ranges[i][0]:ranges[i][1]], sym='')
        axes[0, i].set_xticks(np.arange(1, ranges[i][1] - ranges[i][0] + 1))
        if i == 0:
            axes[0, i].set_xticklabels(base_method_labels, fontsize=8)
        else:
            axes[0, i].set_xticklabels(method_labels, fontsize=8)
        axes[0, i].set(xlabel=scenario_labels[i])
    axes[0, 0].set_ylabel(y_label1, fontsize=8)
    for i in range(nr_scenarios):
        axes[1, i].boxplot(dset2[ranges[i][0]:ranges[i][1]], sym='')
        axes[1, i].set_xticks(np.arange(1, ranges[i][1] - ranges[i][0] + 1))
        if i == 0:
            axes[1, i].set_xticklabels(base_method_labels, fontsize=8, rotation=90)
        else:
            axes[1, i].set_xticklabels(method_labels, fontsize=8, rotation=90)
        axes[1, i].set(xlabel=scenario_labels[i])
    axes[1, 0].set_ylabel(y_label2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_2_var_whisker(var1, var2, tags, path_save, var1_label, var2_label, x_label=None):
    fig, axs = plt.subplots(ncols=2)
    axs[0].boxplot(var1, labels=tags, sym='', widths=0.2)
    axs[0].set_xticks(np.arange(1, len(tags) + 1))
    axs[0].set_xticklabels(tags, fontsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(var1_label, fontsize=8)

    axs[1].boxplot(var2, labels=tags, sym='', widths=0.2)
    axs[1].set_xticks(np.arange(1, len(tags) + 1))
    axs[1].set_xticklabels(tags, fontsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(var2_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def plot_3_var_whisker(var1, var2, var3, tags, path_save, var1_label, var2_label, var3_label, x_label=None):
    fig, axs = plt.subplots(ncols=3)
    axs[0].boxplot(var1, labels=tags, sym='', widths=0.2)
    axs[0].set_xticks(np.arange(1, len(tags) + 1))
    axs[0].set_xticklabels(tags, fontsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(var1_label, fontsize=8)

    axs[1].boxplot(var2, labels=tags, sym='', widths=0.2)
    axs[1].set_xticks(np.arange(1, len(tags) + 1))
    axs[1].set_xticklabels(tags, fontsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(var2_label, fontsize=8)

    var3 = [[t / 60 for t in v] for v in var3]
    axs[2].boxplot(var3, labels=tags, sym='', widths=0.2)
    axs[2].set_xticks(np.arange(1, len(tags) + 1))
    axs[2].set_xticklabels(tags, fontsize=8)
    axs[2].tick_params(axis='y', labelsize=8)
    if x_label:
        axs[2].set_xlabel(x_label)
    axs[2].set_ylabel(var3_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()
    return


def block_trajectories(focus_trip_ids):
    trajectories_in_df = pd.read_csv('in/vis/trajectories_inbound.csv')
    trajectories_out_df = pd.read_csv('in/vis/trajectories_outbound.csv')
    block_info = pd.read_csv('in/vis/block_info.csv')

    block_info = block_info[['trip_id', 'block_id', 'route_type']]
    trajectories_df = pd.concat([trajectories_out_df, trajectories_in_df], axis=0, ignore_index=True)

    trajectories_df = trajectories_df.merge(block_info, on='trip_id')
    focus_blocks = block_info[block_info['trip_id'].isin(focus_trip_ids)]['block_id'].unique()
    trajectories_df = trajectories_df[trajectories_df['block_id'].isin(focus_blocks)]
    trajectories_df['stop_direction'] = trajectories_df['stop_id'].astype(str) + '_' + trajectories_df[
        'route_type'].astype(str)
    terminals = ['8613_1', '386_1', '15136_2', '386_2', '386_0', '8613_0']
    trajectories_df = trajectories_df[trajectories_df['stop_direction'].isin(terminals)]
    trajectories_df['date'] = trajectories_df['avl_arr_time'].str[8:10].astype(int)
    trajectories_df = trajectories_df[
        ['date', 'trip_id', 'block_id', 'stop_sequence', 'stop_id', 'avl_arr_sec', 'avl_dep_sec', 'schd_sec',
         'route_type']]
    trajectories_df = trajectories_df.sort_values(by=['date', 'schd_sec'])
    trajectories_df['avl_arr_sec'] = trajectories_df['avl_arr_sec'] % 86400
    trajectories_df['avl_dep_sec'] = trajectories_df['avl_dep_sec'] % 86400
    unique_dates = trajectories_df['date'].unique()
    fig, axs = plt.subplots(ncols=3, sharex='all', sharey='all')
    i = 0
    for d in [3, 11, 16]:
        trajectories = trajectories_df[trajectories_df['date'] == d]

        trajectories_arr_sec = trajectories[['block_id', 'stop_id', 'avl_arr_sec']]
        trajectories_arr_sec = trajectories_arr_sec.rename(columns={'avl_arr_sec': 'avl_sec'})
        trajectories_dep_sec = trajectories[['block_id', 'stop_id', 'avl_dep_sec']]
        trajectories_dep_sec = trajectories_dep_sec.rename(columns={'avl_dep_sec': 'avl_sec'})
        trajectories_schd_sec = trajectories[['block_id', 'stop_id', 'schd_sec']]
        trajectories_schd_sec = trajectories_schd_sec.rename(columns={'schd_sec': 'avl_sec'})
        trajectories_schd_sec['block_id'] = trajectories_schd_sec['block_id'] * 10

        trajectories_plot = pd.concat([trajectories_arr_sec, trajectories_dep_sec], ignore_index=True)
        trajectories_plot = trajectories_plot.sort_values(by='avl_sec')
        trajectories_plot = trajectories_plot.set_index('avl_sec')
        dict1 = {386: 0, 15136: 20, 8613: 67}
        trajectories_plot = trajectories_plot.replace({'stop_id': dict1})

        trajectories_plot.groupby(by='block_id')['stop_id'].plot(color='red', ax=axs[i])

        trajectories_plot2 = trajectories_schd_sec.sort_values(by='avl_sec')
        trajectories_plot2 = trajectories_plot2.set_index('avl_sec')
        dict1 = {386: 0, 15136: 20, 8613: 67}
        trajectories_plot2 = trajectories_plot2.replace({'stop_id': dict1})
        trajectories_plot2.groupby(by='block_id')['stop_id'].plot(color='silver', alpha=0.5, ax=axs[i])
        axs[i].set_xlim(26500, 34500)
        i += 1
    plt.close()

    fig, axs = plt.subplots(ncols=3, sharex='all', sharey='all')
    trajectories_sim = pd.read_csv('out/trajectories/NC.csv')
    terminals = [386, 8613]
    i = 0

    for rep in [3, 11, 16]:
        trajectories = trajectories_sim[trajectories_sim['replication'] == rep]
        trajectories = trajectories[trajectories['stop_id'].isin(terminals)]

        trajectories_arr_sec = trajectories[['trip_id', 'stop_id', 'arr_sec']]
        trajectories_arr_sec = trajectories_arr_sec.rename(columns={'arr_sec': 'avl_sec'})
        trajectories_dep_sec = trajectories[['trip_id', 'stop_id', 'dep_sec']]
        trajectories_dep_sec = trajectories_dep_sec.rename(columns={'dep_sec': 'avl_sec'})

        trajectories_real = trajectories_df[trajectories_df['date'] == d]
        trajectories_real = trajectories_real[trajectories_real['trip_id'].isin(focus_trip_ids)]
        trajectories_schd_sec = trajectories_real[['trip_id', 'stop_id', 'schd_sec']]
        trajectories_schd_sec = trajectories_schd_sec.rename(columns={'schd_sec': 'avl_sec'})
        # trajectories_schd_sec['trip_id'] = trajectories_schd_sec['trip_id']*10

        trajectories_plot = pd.concat([trajectories_arr_sec, trajectories_dep_sec], ignore_index=True)
        trajectories_plot = trajectories_plot.sort_values(by='avl_sec')
        trajectories_plot = trajectories_plot.set_index('avl_sec')
        dict1 = {386: 0, 15136: 20, 8613: 67}
        trajectories_plot = trajectories_plot.replace({'stop_id': dict1})
        trajectories_plot.groupby(by='trip_id')['stop_id'].plot(color='red', ax=axs[i])

        trajectories_plot2 = trajectories_schd_sec.sort_values(by='avl_sec')
        trajectories_plot2 = trajectories_plot2.set_index('avl_sec')
        dict1 = {386: 0, 15136: 20, 8613: 67}
        trajectories_plot2 = trajectories_plot2.replace({'stop_id': dict1})
        trajectories_plot2.groupby(by='trip_id')['stop_id'].plot(color='silver', alpha=0.5, ax=axs[i])
        axs[i].set_xlim(26500, 34500)
        i += 1
    return
