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
