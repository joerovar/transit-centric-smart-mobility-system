import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from constants import *
from utils import get_geo_stops
plt.rcParams['figure.dpi'] = 120

tstamp_path = 'experiments_0623-1721/'

stops = pd.read_csv(os.path.join(SRC_PATH, SIM_INPUTS_PATH, 'stops.csv'),
                    dtype={'route_id':str})
info = pd.read_csv(
    os.path.join(SRC_PATH, SIM_OUTPUTS_PATH, tstamp_path , 'events.csv'),
    dtype={'route_id':str})
day_str = info['date'].unique()[0]
# temporary
info = info[info['date']==day_str]
info = info.iloc[:201463]
# cut by time
start_step = info[info['time']>=ANIMATION['start_time']]['nr_step'].iloc[0]
end_step = info[info['time']<=ANIMATION['end_time']]['nr_step'].iloc[-1]
tot_frames = end_step-start_step

info = info.merge(stops[['stop_id', 'stop_lon', 'stop_lat']],
                  on='stop_id')

all_stops_xy = {}
ky_stops_xy = {}
for rt in ROUTES:
    x, y, ky_x, ky_y = get_geo_stops(stops, rt,
                                     OUTBOUND_DIRECTIONS[rt])
    all_stops_xy[rt] = {'x': x, 'y': y}
    ky_stops_xy[rt] = {'x': ky_x, 'y': ky_y}
    

load_bounds = ANIMATION['load_bounds']
load_markers = ANIMATION['load_marker_sizes']
dir_colors = ('black', 'gray')
# Define a function to update the scatter plot
def update(frame):
    ax.cla()
    df = info[info['nr_step']==frame].copy()
    for rt in ROUTES:
        df_rt = df[df['route_id']==rt].copy()
        directions = (OUTBOUND_DIRECTIONS[rt], INBOUND_DIRECTIONS[rt])
        ax.plot(all_stops_xy[rt]['x'], all_stops_xy[rt]['y'], color='darkcyan')
        for direction, color in zip(directions, dir_colors):
            df_dir = df_rt[df_rt['direction']==direction].copy()

            if df_dir.shape[0]:
                for i in range(len(load_markers)):
                    lmin, lmax = load_bounds[i], load_bounds[i+1]
                    ldf = df_dir[
                        df_dir['pax_load'].between(lmin, lmax, inclusive='left')].copy()
                    if ldf.shape[0]:
                        scat = ax.scatter(ldf['stop_lon'], ldf['stop_lat'],
                                          color=color, s=load_markers[i],
                                          zorder=3)
        for i in range(len(KEY_STOP_NAMES[rt])):
            plt.annotate(
                KEY_STOP_NAMES[rt][i], (ky_stops_xy[rt]['x'][i], ky_stops_xy[rt]['y'][i]), 
                xytext=(ky_stops_xy[rt]['x'][i], ky_stops_xy[rt]['y'][i]+0.001),
                rotation=40, fontsize=10)
    day_ts = pd.Timestamp(day_str)
    sim_time = df['time'].iloc[0]
    t = (day_ts + pd.to_timedelta(round(sim_time), unit='s')).strftime('%H:%M')
    plt.text(*ANIMATION['text_loc'],t)
    plt.axis('off')
    return scat,

if __name__ == '__main__':
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(7,8))

    df = info[info['nr_step']==start_step].copy()

    for rt in ROUTES:
        df_rt = df[df['route_id']==rt].copy()
        directions = (OUTBOUND_DIRECTIONS[rt], INBOUND_DIRECTIONS[rt])
        ax.plot(all_stops_xy[rt]['x'], all_stops_xy[rt]['y'], color='darkcyan')
        for direction, color in zip(directions, dir_colors):
            df_dir = df_rt[df_rt['direction']==direction].copy()
            if df_dir.shape[0]:
                for i in range(len(load_markers)):
                    lmin, lmax = load_bounds[i], load_bounds[i+1]
                    ldf = df_dir[
                        df_dir['pax_load'].between(lmin, lmax, inclusive='left')].copy()
                    if ldf.shape[0]:
                        scat = ax.scatter(ldf['stop_lon'], ldf['stop_lat'],
                                          color=color, s=load_markers[i],
                                          zorder=3)
        for i in range(len(KEY_STOP_NAMES[rt])):
            plt.annotate(
                KEY_STOP_NAMES[rt][i], (ky_stops_xy[rt]['x'][i], ky_stops_xy[rt]['y'][i]), 
                xytext=(ky_stops_xy[rt]['x'][i], ky_stops_xy[rt]['y'][i]+0.001),
                rotation=40, fontsize=10)
        
    day_ts = pd.Timestamp(day_str)
    sim_time = df['time'].iloc[0]
    t = (day_ts + pd.to_timedelta(round(sim_time), unit='s')).strftime('%H:%M')
    plt.text(*ANIMATION['text_loc'],t)
    plt.axis('off')
    # Create an animation object
    ani = animation.FuncAnimation(fig, update, 
                                  frames=range(start_step,end_step), interval=1)
    ani.save(os.path.join(SRC_PATH, SIM_OUTPUTS_PATH, tstamp_path, 'animation.gif'),
             fps=30)

