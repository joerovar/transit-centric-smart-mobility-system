from input import *


class Trip:
    def __init__(self, trip_id, sched_time, route_type):
        self.trip_id = trip_id
        self.sched_time = sched_time
        self.route_type = route_type
        # this will serve as a copy for pax who finished the trip (main one is an attribute of the simulation class)
        self.completed_pax = []
        # route_type_dict = {0: inbound, 1: outbound long, 2: outbound short}
        # event_dict = {0: 'inbound dispatch', 1: 'inbound intermediate', 2: 'inbound arrival',
        #               3: 'outbound dispatch', 4: 'outbound arrival'}


class Passenger:
    def __init__(self, orig_idx, dest_idx, arr_time):
        self.orig_idx = orig_idx
        self.dest_idx = dest_idx
        self.arr_time = arr_time
        self.board_time = 0.0
        self.alight_time = 0.0
        self.denied = 0
        self.wait_time = 0.0
        self.journey_time = 0.0
        self.trip_id = 0


class Stop:
    def __init__(self, stop_id):
        self.stop_id = stop_id
        self.pax = []
        self.passed_trips = []


class Bus:
    def __init__(self, bus_id, trips_info):
        self.bus_id = bus_id
        self.pending_trips = [Trip(ti[0], ti[1], ti[2]) for ti in trips_info]
        self.active_trip = []
        self.finished_trips = []
        self.next_event_time = 0.0
        self.next_event_type = 0
        self.pax = []
        self.last_stop_id = 0
        self.next_stop_id = 0
        self.arr_t = 0.0
        self.dep_t = 0.0
        self.ons = 0
        self.offs = 0
        self.denied = 0


class Log:
    def __init__(self, trip_identifiers):
        self.recorded_departures = {t: None for t in trip_identifiers}
        self.recorded_arrivals = {t: None for t in trip_identifiers}


class TripLog:
    def __init__(self, trip_id, stops):
        self.trip_id = trip_id
        self.stop_arr_times = {s: None for s in stops}
        self.stop_dep_times = {s: None for s in stops}
