import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

raw_data_dir = "../../data/01_RAW/"
intermediate_data_dir = "../../data/02_INTERMEDIATE/"

gtfs_bus = gpd.read_file(raw_data_dir+"CTA shapefiles/CTA_BusStops.shp").to_crs("EPSG:26916")
gtfs_bus = gtfs_bus[['SYSTEMSTOP','geometry']]

start_date = pd.to_datetime('20190903')
end_date = pd.to_datetime('20190906')

path = "CTA_GTFS_20190903-20191130/"
## STOPS
s = pd.read_csv(raw_data_dir+path+'stops.txt', dtype={'trip_id':str})


## ROUTES
r = pd.read_csv(raw_data_dir+path+'/routes.txt')
t = pd.read_csv(raw_data_dir+path+'/trips.txt', dtype={'service_id':str, 'trip_id':str, 'shape_id':str, 'schd_trip_id':str})
st = pd.read_csv(raw_data_dir+path+'/stop_times.txt', dtype={'trip_id':str})
c = pd.read_csv(raw_data_dir+path+'/calendar.txt', dtype={'service_id':str, 'start_date':str, 'end_date':str})
cd = pd.read_csv(raw_data_dir+path+'/calendar_dates.txt', dtype={'service_id':str, 'date':str})

c['start_date'] = pd.to_datetime(c['start_date'])
c['end_date'] = pd.to_datetime(c['end_date'])
cd['date'] = pd.to_datetime(cd['date'])

# get service_ids from calendar.txt
service_id_all = []
for cur_date in pd.date_range(start_date, end_date):
    print(cur_date)
    dow_col = cur_date.day_name().lower()
    service_ids = c[(c[dow_col] == 1) & (c['start_date']<=cur_date) & (c['end_date']>=cur_date)]['service_id']

    # modify according to calendar_dates.txt
    service_ids = service_ids.append(cd[(cd['date']==cur_date) & (cd['exception_type']==1)]['service_id'])
    service_ids = service_ids[~service_ids.isin(cd[(cd['date']==cur_date) & (cd['exception_type']==2)]['service_id'])]

    service_id_all += service_ids.tolist()

# get ids of scheduled trips from trips.txt
schd_trips = t[t['service_id'].isin(service_id_all)]

avl = pd.read_csv(raw_data_dir+"bus_state_hist_sept_2019.csv", chunksize=10000)

df_list = []
for df in avl:
    df = df[(pd.to_datetime(df['event_time']).dt.date >= datetime.date(2019,9,3)) & \
            (pd.to_datetime(df['event_time']).dt.date <= datetime.date(2019,9,6))]
    df_list.append(df[['bus_state_id', 'event_time', 'event_type', 'bus_id', 'route_id',
       'trip_id', 'stop_id', 'stop_sequence', 'odometer_distance',
       'dwell_time', 'passenger_load', 'ron', 'fon', 'roff', 'foff']])
d = pd.concat(df_list)
d['date'] = pd.to_datetime(d['event_time']).dt.date

for route in ['20']:
    # filter trips to route
    r_schd_trips = schd_trips[schd_trips['route_id']==route]
    # merge in stop information
    r_schd_trip_stops = pd.merge(r_schd_trips, st, on='trip_id')

    r_schd_trip_stops['schd_trip_id'] = r_schd_trip_stops['schd_trip_id'].astype(int)
    r_schd_trip_stops.loc[r_schd_trip_stops['arrival_time'].str[:2]>='24', 'arrival_time'] = \
        r_schd_trip_stops.loc[r_schd_trip_stops['arrival_time'].str[:2]>='24', 'arrival_time'].str[2:]
    r_schd_trip_stops.loc[r_schd_trip_stops['arrival_time'].str.len()==6, 'arrival_time'] = \
        '00'+r_schd_trip_stops.loc[r_schd_trip_stops['arrival_time'].str.len()==6, 'arrival_time']
    r_schd_trip_stops['schd_sec'] = pd.to_timedelta(r_schd_trip_stops['arrival_time']).dt.seconds
    r_schd_trip_stops.loc[r_schd_trip_stops['schd_sec'] < 10800,'schd_sec'] += 86400
    r_schd_trip_stops.sort_values(by=['stop_sequence','schd_sec'],inplace=True)
    r_schd_trip_stops['hour'] = r_schd_trip_stops['schd_sec']//900/4
    r_schd_trip_stops.sort_values(by=['trip_id','stop_id'],inplace=True)


    r_avl = d[(d['route_id']==route)]# & (d['date']==datetime.date(2019,9,4))]
    r_avl = r_avl[r_avl['stop_id'].isin(s['stop_id'])]
    r_avl = r_avl.groupby(['date','trip_id','stop_id'], as_index=False).first()
    r_avl['trip_id'] = r_avl['trip_id'].astype(int)
    r_avl['avl_sec'] = (pd.to_datetime(r_avl['event_time']) - pd.to_datetime('20190904')).dt.total_seconds()
    r_avl.loc[r_avl['avl_sec'] < 10800,'avl_sec'] += 86400
    r_avl.sort_values(by=['trip_id','stop_id'],inplace=True)

    # calculate arrival difference
    merge = pd.merge(r_schd_trip_stops[['route_id','schd_trip_id','stop_id','stop_sequence','arrival_time','schd_sec']], 
                     r_avl[['route_id','trip_id', 'stop_id','event_time','avl_sec']],
                    left_on=['route_id','schd_trip_id','stop_id'],
                    right_on=['route_id','trip_id', 'stop_id'])

    merge['diff'] = merge['avl_sec'] - merge['schd_sec']

merge.to_csv(intermediate_data_dir+"route20_stop_time.csv", index=False)