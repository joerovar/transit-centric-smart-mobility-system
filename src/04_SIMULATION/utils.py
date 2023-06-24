import matplotlib.pyplot as plt
from constants import *
import pandas as pd

def get_geo_stops(stops, rt, direction):
    rt_stops = stops[(stops['route_id']==rt) & 
                     (stops['direction']==direction)].copy()
    shapes = rt_stops.groupby('shape_id')['stop_sequence'].max().reset_index()
    max_stops = shapes['stop_sequence'].max()
    longest_shape = shapes.loc[shapes['stop_sequence']==max_stops, 'shape_id'].iloc[0]
    rt_stops = rt_stops[rt_stops['shape_id']==longest_shape]
    rt_stops = rt_stops.sort_values(by='stop_sequence').reset_index(drop=True)

    lon, lat = rt_stops['stop_lon'].tolist(), rt_stops['stop_lat'].tolist()

    key_stops = rt_stops[
    (rt_stops['stop_name'].str.contains('|'.join(KEY_STOP_NAMES[rt])))
    ].copy()
    if rt == '92':
        key_stops = key_stops[key_stops['stop_id']!=15120]
    key_lon, key_lat = key_stops['stop_lon'].tolist(), key_stops['stop_lat'].tolist()
    return  lon, lat, key_lon, key_lat

def plot_situation(situation_df, stops, day_str):
    fig, ax = plt.subplots(figsize=(10,13))
    df = situation_df.copy()
    df = df.merge(stops[['stop_id', 'stop_lon', 'stop_lat']], on='stop_id')
    dir_colors = ('black', 'gray')
    load_bounds = ANIMATION['load_bounds']
    load_alphas = ANIMATION['load_alphas']
    load_markers = ANIMATION['load_marker_sizes']
    for rt in ROUTES:
        df_rt = df[df['route_id']==rt].copy()
        directions = (OUTBOUND_DIRECTIONS[rt], INBOUND_DIRECTIONS[rt])
        x, y, ky_x, ky_y = get_geo_stops(stops, rt, OUTBOUND_DIRECTIONS[rt])
        ky_stops_xy = {'x': ky_x, 'y': ky_y}
        scat = ax.scatter(ky_x, ky_y, marker='+', color='darkcyan')
        ax.plot(x, y, color='darkcyan')

        for direction, color in zip(directions, dir_colors):
            df_dir = df_rt[df_rt['direction']==direction].copy()

            if df_dir.shape[0]:
                for i in range(len(load_alphas)):
                    lmin, lmax = load_bounds[i], load_bounds[i+1]
                    ldf = df_dir[
                        df_dir['pax_load'].between(lmin, lmax, inclusive='left')].copy()
                    if ldf.shape[0]:
                        scat = ax.scatter(ldf['stop_lon'], ldf['stop_lat'],
                                          color=color, s=load_markers[i],
                                          zorder=3)

        for i in range(len(KEY_STOP_NAMES[rt])):
            plt.annotate(
                KEY_STOP_NAMES[rt][i], (ky_stops_xy['x'][i], ky_stops_xy['y'][i]), 
                xytext=(ky_stops_xy['x'][i], ky_stops_xy['y'][i]+0.001),
                rotation=40, fontsize=10)
            
    # ax.set_ylim(*ANIMATION['y_lim'])
    # ax.set_xlim(*ANIMATION['x_lim'])
    day_ts = pd.Timestamp(day_str)
    sim_time = df['time'].iloc[0]
    t = (day_ts + pd.to_timedelta(round(sim_time), unit='s')).strftime('%H:%M')
    plt.text(*ANIMATION['text_loc'],t)
    plt.axis('off')
        
def get_results(exp_tstamp, scenarios, start_t, end_t):
    exp_path = 'experiments_' + exp_tstamp
    pax = []
    trips = []
    for sc in scenarios:
        path_results = os.path.join(SIM_OUTPUTS_PATH, exp_path, sc)
        pax.append(pd.read_csv(os.path.join(path_results, 'pax.csv')))
        trips.append(pd.read_csv(os.path.join(path_results, 'trips.csv')))

    results = {
        'metric': ['wait_t', 'rbt', 'denied', 'sd_load_pk_e', 'run_time_e_95', 'run_time_w_95'],
        'EHD': [],
        'NC': []
    }

    for pax, method in zip(pax, scenarios):
        p = pax[(pax['arrival_time']>=start_t) & 
                (pax['arrival_time']<=end_t)].copy()

        p['wt'] = p['boarding_time'] - p['arrival_time']
        p['jt'] = p['alighting_time'] - p['arrival_time']

        count = p.groupby(['boarding_stop', 'alighting_stop'])['wt'].count().reset_index()
        count = count.rename(columns={'wt':'count'})
        avg = p.groupby(['boarding_stop', 'alighting_stop'])['wt'].mean().reset_index()
        count_avg = avg.merge(count, on=['boarding_stop', 'alighting_stop'])
        weighted = count_avg['count'] * count_avg['wt']
        weighted_avg = weighted.sum() / count_avg['count'].sum()
        results[method].append(weighted_avg)

        count = p.groupby(['boarding_stop', 'alighting_stop'])['jt'].count().reset_index()
        count = count.rename(columns={'jt':'count'})
        median = p.groupby(['boarding_stop', 'alighting_stop'])['jt'].mean().reset_index()
        prc95 = p.groupby(['boarding_stop', 'alighting_stop'])['jt'].quantile(0.95).reset_index()
        rbt = prc95['jt'] - median['jt']
        rbt_count = count.copy()
        rbt_count['rbt'] = rbt
        weighted = rbt_count['count'] * rbt_count['rbt']
        weighted_avg = weighted.sum() / rbt_count['count'].sum()
        results[method].append(weighted_avg)

        count = p['times_denied'].sum()
        results[method].append(count)
    
    for trips, method in zip(trips, scenarios):
        t = trips[(trips['arrival_time_sec']>=start_t) &
                (trips['arrival_time_sec']<=end_t)].copy()
        
        sd = t.loc[t['stop_sequence']==20, 'passenger_load'].std()
        results[method].append(sd)

        dep_ts = t.groupby(['trip_id', 'direction'])['departure_time_sec'].min().reset_index()
        arr_ts = t.groupby(['trip_id', 'direction'])['arrival_time_sec'].max().reset_index()
        run_ts = dep_ts.merge(arr_ts, on=['trip_id', 'direction'])
        run_ts['run_time'] = run_ts['arrival_time_sec'] - run_ts['departure_time_sec']

        run_ts_95 = run_ts.groupby('direction')['run_time'].quantile(0.95).reset_index()

        results[method].append(run_ts_95.loc[run_ts_95['direction']=='East', 'run_time'].values[0])
        results[method].append(run_ts_95.loc[run_ts_95['direction']=='West', 'run_time'].values[0])
    res_df = pd.DataFrame(results)
    res_df['prc_imp'] = ((res_df['EHD'] - res_df['NC'])/res_df['NC']*100).round(2)
    return res_df
