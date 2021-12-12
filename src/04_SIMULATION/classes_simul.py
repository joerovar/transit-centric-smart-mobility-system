from input import *


class Trip:
    def __init__(self, trip_id):
        self.trip_id = trip_id
        self.pax = []


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



