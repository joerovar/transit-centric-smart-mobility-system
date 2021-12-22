import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from datetime import timedelta
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join
import post_process
from scipy.stats import lognorm, norm
# from classes_simul import Passenger, Stop, Trip
import seaborn as sn
from post_process import *


class Trip:
    def __init__(self, trip_id, route_type):
        self.trip_id = trip_id
        self.route_type = route_type
        self.pax = []


class Bus:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.pending_trips = []
        self.active_trip = []
        self.finished_trips = []
        self.next_event_time = 0.0
        self.next_event_type = 0


buses = []
for i in range(5):
    buses.append(Bus(i))
    for j in range(3):
        buses[-1].pending_trips.append(Trip(j, 1))
bus = buses[0]
print([buses[0].pending_trips, buses[0].active_trip, buses[0].finished_trips])
bus.active_trip.append(bus.pending_trips[0])
bus.pending_trips.pop(0)
print([buses[0].pending_trips, buses[0].active_trip, buses[0].finished_trips])
bus.finished_trips.append(bus.active_trip[0])
bus.active_trip.pop(0)
print([buses[0].pending_trips, buses[0].active_trip, buses[0].finished_trips])

bus1 = buses[1]
bus2 = buses[2]
bus1.active_trip.append(bus1.pending_trips[0])
bus1.pending_trips.pop(0)
bus1.next_event_time = 20
bus2.active_trip.append(bus2.pending_trips[0])
bus2.pending_trips.pop(0)
bus2.next_event_time = 12

active_buses = [b for b in buses if b.active_trip]
active_neighbor = [b for b in buses if b.active_trip and b != bus1]
print(active_neighbor)
next_bus = min(active_buses, key=lambda b: b.next_event_time)
print(buses)
print(next_bus.bus_id)
print(active_buses)
print(next_bus)

bus = buses[1]
trajectories = []
trajectories.append([bus.bus_id, bus.next_event_time])
bus = buses[2]
trajectories.append([bus.bus_id, bus.next_event_time])
print(trajectories)
