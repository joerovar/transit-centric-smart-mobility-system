"""
Created on Mon Aug 16 14:02:47 2021

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


survey = pd.read_csv(raw_data_dir+"cta_survey_trips.csv")
s1 = gpd.GeoDataFrame(survey,
    geometry=gpd.points_from_xy(survey.longitude_1, survey.latitude_1))
s2 = gpd.GeoDataFrame(survey,
    geometry=gpd.points_from_xy(survey.longitude_2, survey.latitude_2))

org_in = s1.geometry.intersects(bus_line.iloc[0].geometry)
des_in = s2.geometry.intersects(bus_line.iloc[0].geometry)


survey_route = survey[org_in | des_in]

survey_route['mode'] = survey_route['mode_2']//100
survey_route['mode'] = survey_route['mode'].map({1:1,2:2,3:3,4:3,5:4,6:3,7:3})
survey_route_mode = survey_route['mode'].value_counts().sort_index() / len(survey_route)
survey_route_mode = survey_route_mode.rename({1: 'Active', 2: 'Auto', 3: 'Mobility Services', 4: 'Public Transit'})
survey_route_mode.to_csv(intermediate_data_dir+"mode_choice.csv")
print("1: Active; 2: Auto; 3: Mobility Services; 4: Public Transit")
print(survey_route['mode'].value_counts().sort_index() )


