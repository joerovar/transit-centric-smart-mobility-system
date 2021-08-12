import geopandas as gpd
import numpy as np
import pandas as pd
import datetime

data_dir = "../../data/01_RAW/"
intermediate_data_dir = "../../data/02_INTERMEDIATE"

data = pd.read_csv(data_dir+"cta_odx_journeys_0902_06_2019.csv", chunksize=10000)

df_list = []
for df in data:
    # Not labor day
    df = df[pd.to_datetime(df['SERVICE_DT']).dt.date != datetime.date(2019,9,2)]
    # 7am - 10am
    df['minute'] = pd.to_datetime(df['TRANSACTION_DTM']).dt.hour*60+pd.to_datetime(df['TRANSACTION_DTM']).dt.minute
    df['bin_5'] = df['minute'] // 5
    df = df[(df['bin_5']>=84)&(df['bin_5']<120)]
    # Route 20
    df = df[df['AVL_BUS_ROUTE']=='20'] 
    
    df_list.append(df[['TRANSACTION_DTM','OPERATOR_NM','BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP','AVL_BUS_ROUTE','bin_5']])

d = pd.concat(df_list)

d.loc[:,'servicedate'] = pd.to_datetime(d['TRANSACTION_DTM']).dt.date

# Take out observations where destination cannot be inferred
od = d.dropna()
print("Caution: %.2f%% ridership left." % (len(od)/len(d)*100))

all_od = od[['BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP']].drop_duplicates()
all_dates = od[['servicedate','bin_5']].drop_duplicates().sort_values(by=['servicedate','bin_5'])
all_od['key'] = 1
all_dates['key'] = 1
all_comb = pd.merge(all_od,all_dates,on='key')

# count trips
trip_cnt = od.groupby(['servicedate','BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP','bin_5'], as_index=False).count()
# fill in days where no trips recorded for the OD
trip_cnt = pd.merge(all_comb, trip_cnt, on=['servicedate','BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP','bin_5'], how='left').fillna(0)
# calculate mean and std
trip_cnt = trip_cnt.groupby(['BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP','bin_5'], as_index=False).agg({'TRANSACTION_DTM':['mean','std']})
# rename columns
trip_cnt.columns = ['BOARDING_STOP','INFERRED_ALIGHTING_GTFS_STOP','bin_5','mean','std']

trip_cnt.to_csv(intermediate_data_dir + "odt_for_opt.csv", index=False)