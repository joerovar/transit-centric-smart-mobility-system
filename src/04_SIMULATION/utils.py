import matplotlib.pyplot as plt
from constants import *
import pandas as pd

# Create a figure and axis object
def plot_situation(situation_df, stops, key_stops):

    lon = stops['stop_lon'].tolist()
    lat = stops['stop_lat'].tolist()

    key_lon = key_stops['stop_lon'].tolist()
    key_lat = key_stops['stop_lat'].tolist()
    key_stops_coord = list(zip(key_lon, key_lat))

    fig, ax = plt.subplots(figsize=(9,5))

    df = situation_df.copy()
    east_df = df[df['direction']=='East'].copy()
    west_df = df[df['direction']=='West'].copy()
    if east_df.shape[0]:
        scat = ax.scatter(east_df['stop_lon'], east_df['stop_lat'], color='black')
    if west_df.shape[0]:
        scat = ax.scatter(west_df['stop_lon'], west_df['stop_lat'], color='gray')
    scat = ax.scatter(key_lon, key_lat, marker='|', color='darkcyan')
    ax.plot(lon, lat, color='darkcyan')
    ax.set_ylim(*ANIMATION['y_lim'])
    ax.set_xlim(*ANIMATION['x_lim'])
    t = (pd.Timestamp.today().floor('D') + pd.to_timedelta(df['time'].iloc[0], unit='s')).strftime('%H:%M')
    plt.text(*ANIMATION['text_loc'],t)
    plt.axis('off')
    for i in range(len(KEY_STOP_NAMES)):
        plt.annotate(
            KEY_STOP_NAMES[i], key_stops_coord[i], 
            xytext=(key_stops_coord[i][0], key_stops_coord[i][1]+0.001),
            rotation=60, fontsize=10)
        
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
