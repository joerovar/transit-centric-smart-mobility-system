class Trip:
    def __init__(self, trip_id, route_type, stops, schedule, dist_traveled):
        self.trip_id = trip_id
        self.direction = route_type
        self.stops = stops
        self.schedule = schedule
        self.dist_traveled = dist_traveled
        # this will serve as a copy for pax who finished the trip (main one is an attribute of the simulation class)
        self.completed_pax = []
        # route_type_dict = {0: outbound, 1: inbound}
        # event_dict = {0: 'outbound dispatch', 1: 'outbound intermediate', 2: 'outbound arrival',
        #               3: 'inbound dispatch', 4: 'inbound arrival'}


class Passenger:
    def __init__(self, orig_id, dest_id, arr_time):
        self.orig_id = orig_id
        self.dest_id = dest_id
        self.arr_time = arr_time
        self.board_time = 0.0
        self.alight_time = 0.0
        self.denied = 0
        self.trip_id = 0


class Stop:
    def __init__(self, stop_id, sched_t):
        self.stop_id = stop_id
        self.sched_t = sched_t
        self.pax = []
        self.passed_trips = []
        self.last_arr_t = []
        self.last_dep_t = []


class Bus:
    def __init__(self, bus_id, trips_info):
        self.bus_id = bus_id
        self.pending_trips = [Trip(ti[0], ti[1], ti[2], ti[3], ti[4]) for ti in trips_info]
        self.active_trip = []
        self.finished_trips = []
        self.next_event_time = float('nan')
        self.next_event_type = float('nan')
        self.pax = []
        self.last_stop_id = None
        self.next_stop_id = None
        self.arr_t = float('nan')
        self.dep_t = float('nan')
        self.ons = 0
        self.offs = 0
        self.denied = 0
        self.cancelled = False
        self.cancelled_trips = []
        self.instructed_hold_time = None # at terminal
        self.expressed = False

    def deactivate(self):
        # assert not self.pax
        self.next_event_time = float('nan')
        self.next_event_type = float('nan')
        self.last_stop_id = None
        self.next_stop_id = None
        self.arr_t = float('nan')
        self.dep_t = float('nan')
        self.ons = float('nan')
        self.offs = float('nan')
        self.denied = float('nan')
        self.cancelled = False
        self.cancelled_trips = []
        self.instructed_hold_time = None # at terminal
        self.expressed = False


class TripLog:
    def __init__(self, trip_id, stops):
        self.trip_id = trip_id
        self.stop_arr_times = {s: None for s in stops}
        self.stop_dep_times = {s: None for s in stops}

