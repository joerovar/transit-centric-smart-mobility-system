"""
Created on Mon Aug 16 15:32:39 2021

@author: qingyi
"""

import geopandas as gpd
import geoplot as gplt
import numpy as np
import pandas as pd

raw_data_dir = "../../data/01_RAW/"
intermediate_data_dir = "../../data/02_INTERMEDIATE/"

bus_line = gpd.read_file(raw_data_dir+"CTA shapefiles/CTA_BusRoutes.shp")
bus_line = bus_line[bus_line['ROUTE']=='20']
bus_line = bus_line.to_crs("EPSG:26916")
bus_line.geometry = bus_line.geometry.buffer(500)
bus_line = bus_line.to_crs("EPSG:4326")


poi = pd.read_csv(raw_data_dir+"chicago_safegraph_poi.csv")
poi = gpd.GeoDataFrame(poi,
    geometry=gpd.points_from_xy(poi.sg_c__longitude, poi.sg_c__latitude))
poi['naics_top2'] = poi['sg_c__naics_code']//10000


naics_top2_group_names = {
23: "Construction and Mining",
44: "Retail and Wholesale Trade",
48: "Transportation and Warehousing",
51: "Coporation",
71: "Recreation",
72: "Accomodation and Food",
81: "Other",
92: "Public Admin"}

poi['naics_top2_group'] = poi['naics_top2'].map(
    {23:23,31:23,32:23,33:23,42:44,
     44:44,45:44,48:48,49:48,51:51,
     52:51,53:51,54:51,55:51,56:51,
     61:51,62:51,71:71,72:72,81:81,92:92})

poi_all = poi.groupby('naics_top2_group', as_index=False).count()[['naics_top2_group','placekey']]
poi_all['poi_prop'] = poi_all['placekey']/poi_all['placekey'].sum()

poi_in = poi[poi.geometry.intersects(bus_line.iloc[0].geometry)].copy()
poi_in = poi_in.groupby('naics_top2_group', as_index=False).count()[['naics_top2_group','placekey']]
poi_in['poi_prop'] = poi_in['placekey']/poi_in['placekey'].sum()


poi_prop = pd.merge(poi_in, poi_all, on='naics_top2_group', suffixes=("_in","_all"))
poi_prop['poi_name'] = poi['naics_top2_group'].map(naics_top2_group_names)

poi_prop.to_csv(intermediate_data_dir+"route20_poi.csv", index=False)
