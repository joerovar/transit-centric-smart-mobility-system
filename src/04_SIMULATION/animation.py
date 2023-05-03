import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from constants import *
plt.rcParams['figure.dpi'] = 140

tstamp_path = 'experiments_0502-1702/EHD'

info_records = pd.read_csv(
    os.path.join(SRC_PATH, SIM_OUTPUTS_PATH, tstamp_path , 'events.csv'))
info_records = info_records[info_records['active']==1]

# get east stops for drawing line
schedule = pd.read_csv(os.path.join(SRC_PATH, SIM_INPUTS_PATH, 'schedule.csv'))
stops_east = pd.read_csv(os.path.join(SRC_PATH, SIM_INPUTS_PATH, 'stops_outbound.csv'))


east_trip = schedule.loc[schedule['direction']=='East', 'trip_id'].iloc[0]
east_trip_sched = schedule[schedule['trip_id']==east_trip].copy()
stops_east['stop_id'] = stops_east['stop_id'].astype(int)
east_trip_sched = east_trip_sched.merge(stops_east[[
    'stop_id', 'stop_lon', 'stop_lat', 'stop_name']], on='stop_id')
stops_e = east_trip_sched.sort_values(by='stop_sequence')[['stop_lon', 'stop_lat']]
stops_lon = stops_e['stop_lon'].tolist()
stops_lat = stops_e['stop_lat'].tolist()
key_stops_e = east_trip_sched.sort_values(by='stop_sequence')
key_stops_e = key_stops_e[
    key_stops_e['stop_name'].str.contains('|'.join(KEY_STOP_NAMES))
].copy()

key_stops_e_lon = key_stops_e['stop_lon'].tolist()
key_stops_e_lat = key_stops_e['stop_lat'].tolist()
key_stops_coord = list(zip(key_stops_e_lon, key_stops_e_lat))

# crs = {'init': 'epsg:4326'} # specify the epsg code for the crs

# Define a function to update the scatter plot
def update(frame):
    ax.cla()
    df = info_records[info_records['nr_step']==frame].copy()
    east_df = df[df['direction']=='East'].copy()
    west_df = df[df['direction']=='West'].copy()
    if east_df.shape[0]:
        for load_min, load_max, alpha in zip([0,10,30], [10, 30, 70], [0.4, 0.7, 1.0]):
            edf = east_df[(east_df['pax_load'] >= load_min) & (east_df['pax_load'] < load_max)].copy()
            if edf.shape[0]:
                scat = ax.scatter(edf['stop_lon'], edf['stop_lat'], color='black', alpha=alpha)
        # scat = ax.scatter(east_df['stop_lon'], east_df['stop_lat'], color='black')
    if west_df.shape[0]:
        for load_min, load_max, alpha in zip([0,10,30], [10, 30, 70], [0.4, 0.7, 1.0]):
            wdf = west_df[(west_df['pax_load'] >= load_min) & (west_df['pax_load'] < load_max)].copy()
            if wdf.shape[0]:
                scat = ax.scatter(wdf['stop_lon'], wdf['stop_lat'], color='gray', alpha=alpha)
    scat = ax.scatter(key_stops_e_lon, key_stops_e_lat, marker='|', color='darkcyan')
    ax.plot(stops_lon, stops_lat, color='darkcyan')
    ax.set_ylim(*ANIMATION['y_lim'])
    ax.set_xlim(*ANIMATION['x_lim'])
    t = (pd.Timestamp.today().floor('D') + pd.to_timedelta(df['time'].iloc[0], unit='s')).strftime('%H:%M')
    plt.text(*ANIMATION['text_loc'],t)
    plt.axis('off')
    for i in range(len(KEY_STOP_NAMES)):
        plt.annotate(
            KEY_STOP_NAMES[i], key_stops_coord[i], 
            xytext=(key_stops_coord[i][0], key_stops_coord[i][1]+0.001),
            rotation=60, fontsize=8)
        
    return scat,

if __name__ == '__main__':
    # Create a figure and axis object
    fig, ax = plt.subplots()

    df = info_records[info_records['nr_step']==499].copy()
    east_df = df[df['direction']=='East'].copy()
    west_df = df[df['direction']=='West'].copy()
    if east_df.shape[0]:
        for load_min, load_max, alpha in zip([0,10,30], [10, 30, 70], [0.4, 0.7, 1.0]):
            edf = east_df[(east_df['pax_load'] >= load_min) & (east_df['pax_load'] < load_max)].copy()
            if edf.shape[0]:
                scat = ax.scatter(edf['stop_lon'], edf['stop_lat'], color='black', alpha=alpha)

    if west_df.shape[0]:
        for load_min, load_max, alpha in zip([0,10,30], [10, 30, 70], [0.4, 0.7, 1.0]):
            wdf = west_df[(west_df['pax_load'] >= load_min) & (west_df['pax_load'] < load_max)].copy()
            if wdf.shape[0]:
                scat = ax.scatter(wdf['stop_lon'], wdf['stop_lat'], color='gray', alpha=alpha)

    scat = ax.scatter(key_stops_e_lon, key_stops_e_lat, marker='|', color='darkcyan')
    ax.plot(stops_lon, stops_lat, color='darkcyan')
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
        
    # Create an animation object
    ani = animation.FuncAnimation(fig, update, frames=range(500,2500), interval=1)
    ani.save(os.path.join(SRC_PATH, SIM_OUTPUTS_PATH, tstamp_path, 'animation.gif'))

