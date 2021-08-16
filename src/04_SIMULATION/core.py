from input import *
import numpy as np
from scipy.stats import lognorm
import random


def get_interval(time):
    interval = int(time / (INTERVAL_LENGTH_MINS * 60))
    return interval


class SimulationEnv:
    def __init__(self):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        # SIMUL PARAMS
        # NEW
        self.next_departures = []
        self.next_trip_ids = []
        self.active_trips = []
        self.last_stop = []
        self.next_stop = []
        self.dep_t = []
        self.terminal_dep_t = []
        self.arr_t = []
        self.load = []
        self.bus_idx = 0
        self.event_type = 0
        self.next_instance_time = []
        self.time = 0.0
        self.last_bus_time = {}
        # RECORDINGS
        self.trajectories = []
        self.recorded_headway = {}
        self.recorded_trip_times = []

    def create_trip(self):
        trip_file = [
            self.next_instance_time, self.last_stop, self.next_stop, self.load,
            self.dep_t, self.arr_t, self.terminal_dep_t
        ]
        for tf in trip_file:
            tf.append(0)
        return

    def remove_trip(self):
        trip_file = [
            self.active_trips, self.next_instance_time, self.last_stop, self.next_stop, self.load,
            self.dep_t, self.arr_t, self.terminal_dep_t
        ]
        for tf in trip_file:
            tf.pop(self.bus_idx)
        return

    def get_pickups_start(self, headway):
        i = self.bus_idx
        stop = self.last_stop[i]
        arrival_rates = ARRIVAL_RATES[stop]
        inter = get_interval(self.time)
        e_pax = arrival_rates[inter - START_INTERVAL] * headway/60
        pax_at_stop = np.random.poisson(lam=e_pax)
        return pax_at_stop

    def get_pickups(self, headway, last_arrival_t):
        i = self.bus_idx
        stop = self.last_stop[i]
        arrival_rates = ARRIVAL_RATES[stop]
        e_pax = 0
        t = last_arrival_t
        t2 = last_arrival_t + headway
        while t < t2:
            inter = get_interval(t)
            edge = (inter + 1) * 60 * INTERVAL_LENGTH_MINS
            e_pax += arrival_rates[inter - START_INTERVAL] * ((min(edge, t2) - t)/60)
            t = min(edge, t2)
        pax_at_stop = np.random.poisson(lam=e_pax)
        return pax_at_stop

    def get_dropoffs(self):
        i = self.bus_idx
        interv = get_interval(self.time)
        interv_idx = interv - START_INTERVAL
        dropoffs = int(round(ALIGHT_FRACTIONS[self.last_stop[i]][interv_idx] * self.load[i]))
        return dropoffs

    def get_travel_time(self):
        i = self.bus_idx
        prev_stop = self.last_stop[i]
        next_stop = self.next_stop[i]
        mean_runtime = LINK_TIMES_MEAN[str(prev_stop)+'-'+str(next_stop)]
        s = LOGN_S
        runtime = lognorm.rvs(s, scale=mean_runtime)
        assert TTD == 'LOGNORMAL'
        return runtime

    def record_trajectories(self):
        i = self.bus_idx
        trip_id = self.active_trips[i]
        # IN THE END I WANT IT LIKE THIS
        if self.event_type == 2:
            trajectory = [self.next_stop[i], round(self.arr_t[i], 2), 0]
            self.trajectories[trip_id].append(trajectory)
        if self.event_type == 1:
            trajectory = [self.last_stop[i], round(self.dep_t[i], 2), self.load[i]]
            self.trajectories[trip_id].append(trajectory)
        if self.event_type == 0:
            trajectory = [self.last_stop[i], round(self.dep_t[i], 2), self.load[i]]
            self.trajectories[trip_id].append(trajectory)
        return

    def fixed_stop_arrival(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.next_stop[i])
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.next_stop[i] = STOPS[curr_stop_idx + 1]
        dropoffs = self.get_dropoffs()
        # dropoffs = int(round(ALIGHT_FRACTIONS[self.last_stop[i]] * self.load[i]))
        self.load[i] -= dropoffs
        assert self.load[i] >= 0
        if self.last_bus_time[self.last_stop[i]] == START_SIMUL_TIME:
            headway = INIT_HEADWAY
            pickups = self.get_pickups_start(headway)
        else:
            headway = self.time - self.last_bus_time[self.last_stop[i]]
            if headway < 0:
                headway = 0
            self.recorded_headway[self.last_stop[i]].append(headway)
            pickups = self.get_pickups(headway, self.last_bus_time[self.last_stop[i]])
        dwell_time = round(STOPPING_DELAY + pickups * BOARDING_DELAY + dropoffs * ALIGHTING_DELAY, 1)
        dwell_time = (pickups + dropoffs > 0) * dwell_time
        self.load[i] += pickups
        self.dep_t[i] = self.time + dwell_time
        self.last_bus_time[self.last_stop[i]] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        self.record_trajectories()
        return

    def terminal_arrival(self):
        i = self.bus_idx
        self.arr_t[i] = self.time
        self.record_trajectories()
        if self.last_bus_time[self.next_stop[i]] == START_SIMUL_TIME:
            headway = INIT_HEADWAY
        else:
            headway = self.time - self.last_bus_time[self.next_stop[i]]
            if headway < 0:
                headway = 0
            self.recorded_headway[self.next_stop[i]].append(headway)
        actual_route_time = self.time - self.terminal_dep_t[i]
        self.recorded_trip_times.append(actual_route_time)
        self.last_bus_time[self.next_stop[i]] = self.time
        # delete record of trip
        self.remove_trip()
        return

    def terminal_departure(self):
        # first create file for trip
        i = self.bus_idx
        self.create_trip()
        self.last_stop[i] = STOPS[0]
        self.next_stop[i] = STOPS[1]
        self.dep_t[i] = self.time
        self.terminal_dep_t[i] = self.time
        if self.time == START_SIMUL_TIME:
            headway = INIT_HEADWAY
            pickups = self.get_pickups_start(headway)
        else:
            headway = self.dep_t[i] - self.last_bus_time[self.last_stop[i]]
            self.recorded_headway[self.last_stop[i]].append(headway)
            pickups = self.get_pickups(headway, self.last_bus_time[self.last_stop[i]])
        self.load[i] += pickups
        self.record_trajectories()
        self.last_bus_time[self.last_stop[i]] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        return

    def next_event(self):
        if self.next_departures:
            if False not in [self.next_departures[0] < n for n in self.next_instance_time]:
                self.time = self.next_departures[0]
                self.active_trips.append(self.next_trip_ids[0])
                self.bus_idx = -1
                self.event_type = 0
                self.next_departures.pop(0)
                self.next_trip_ids.pop(0)
                return
        if self.next_instance_time:
            self.time = min(self.next_instance_time)
            self.bus_idx = self.next_instance_time.index(self.time)
            next_stop = self.next_stop[self.bus_idx]
            assert next_stop != STOPS[0]
            if next_stop == STOPS[-1]:
                self.event_type = 2
            else:
                self.event_type = 1
            return

    def reset_simulation(self):
        self.next_departures = SCHEDULED_DEPARTURES.copy()
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(self.next_departures))]
        self.next_departures = [sum(x) for x in zip(self.next_departures, dep_delays)]
        self.next_trip_ids = [i for i in range(1, 1 + len(SCHEDULED_DEPARTURES))]
        self.time = START_SIMUL_TIME
        self.bus_idx = 0

        # trip-level data
        self.next_instance_time = []
        self.active_trips = []
        self.last_stop = []
        self.next_stop = []
        self.load = []
        self.dep_t = []
        self.arr_t = []
        self.terminal_dep_t = []
        self.event_type = 0

        # stop-level data
        for s in STOPS:
            self.recorded_headway[s] = []
            self.last_bus_time[s] = self.time

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []
        self.recorded_trip_times = []
        return False

    def prep(self):
        self.next_event()
        if self.time >= STOP_SIMUL_TIME:
            return True
        if self.event_type == 2:
            self.terminal_arrival()
            return not self.next_departures and not self.next_instance_time

        if self.event_type == 1:
            self.fixed_stop_arrival()
            return self.prep()

        if self.event_type == 0:
            self.terminal_departure()
            return self.prep()

