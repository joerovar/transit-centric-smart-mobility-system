from input import *
import numpy as np
from scipy.stats import lognorm
import random


def get_interval(t, interval_length):
    interval = int(t / (interval_length * 60))
    return interval

class SimulationEnv:
    def __init__(self, no_overtake_policy=True):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        self.no_overtake_policy = no_overtake_policy
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
        self.track_denied_boardings = {}
        # RECORDINGS
        self.trajectories = {}

    def record_trajectories(self, pickups=0, dropoffs=0, denied_board=0):
        i = self.bus_idx
        trip_id = self.active_trips[i]
        # IN THE END I WANT IT LIKE THIS
        if self.event_type == 2:
            trajectory = [self.next_stop[i], round(self.arr_t[i], 2), 0, pickups, dropoffs, denied_board]
            self.trajectories[trip_id].append(trajectory)
        if self.event_type == 1:
            trajectory = [self.last_stop[i], round(self.dep_t[i], 2), self.load[i], pickups, dropoffs, denied_board]
            self.trajectories[trip_id].append(trajectory)
        if self.event_type == 0:
            trajectory = [self.last_stop[i], round(self.dep_t[i], 2), self.load[i], pickups, dropoffs, denied_board]
            self.trajectories[trip_id].append(trajectory)
        return

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

    def get_arrivals_start(self, headway):
        i = self.bus_idx
        stop = self.last_stop[i]
        arrival_rates = ARRIVAL_RATES[stop]
        inter = get_interval(self.time, DEM_INTERVAL_LENGTH_MINS)
        e_pax = (arrival_rates[inter - DEM_START_INTERVAL]/60) * headway/60
        pax_at_stop = np.random.poisson(lam=e_pax)
        return pax_at_stop

    def get_pax_arrivals(self, headway, last_arrival_t):
        i = self.bus_idx
        stop = self.last_stop[i]
        arrival_rates = ARRIVAL_RATES[stop]
        e_pax = 0
        t = last_arrival_t
        t2 = last_arrival_t + headway
        while t < t2:
            inter = get_interval(t, DEM_INTERVAL_LENGTH_MINS)
            edge = (inter + 1) * 60 * DEM_INTERVAL_LENGTH_MINS
            e_pax += (arrival_rates[inter - DEM_START_INTERVAL]/60) * ((min(edge, t2) - t)/60)
            t = min(edge, t2)
        pax_arrivals = np.random.poisson(lam=e_pax)
        return pax_arrivals

    def get_dropoffs(self):
        i = self.bus_idx
        interv = get_interval(self.time, DEM_INTERVAL_LENGTH_MINS)
        interv_idx = interv - DEM_START_INTERVAL
        dropoffs = int(round(ALIGHT_FRACTIONS[self.last_stop[i]][interv_idx] * self.load[i]))
        return dropoffs

    def get_travel_time(self):
        i = self.bus_idx
        prev_stop = self.last_stop[i]
        next_stop = self.next_stop[i]
        interv = get_interval(self.time, TIME_INTERVAL_LENGTH_MINS)
        interv_idx = interv - TIME_START_INTERVAL
        mean_runtime = LINK_TIMES_MEAN[str(prev_stop)+'-'+str(next_stop)][interv_idx]
        assert mean_runtime
        s = LOGN_S
        runtime = lognorm.rvs(s, scale=mean_runtime)
        assert TTD == 'LOGNORMAL'
        return runtime

    def no_overtake(self):
        # if this is the first bus then ignore and return
        curr_bus_idx = self.bus_idx
        next_instance_t = self.next_instance_time[curr_bus_idx]
        if curr_bus_idx == -1:
            curr_bus_idx = self.active_trips.index(self.active_trips[curr_bus_idx])
        if curr_bus_idx:
            lead_bus_idx = curr_bus_idx-1
            try:
                lead_bus_next_stop = self.next_stop[lead_bus_idx]
            except IndexError:
                print(curr_bus_idx)
                print(lead_bus_idx)
                print(self.last_stop)
                raise
            curr_bus_next_stop = self.next_stop[curr_bus_idx]
            if lead_bus_next_stop == curr_bus_next_stop:
                prev_bus_next_instance = self.next_instance_time[lead_bus_idx]
                next_instance_t = max(prev_bus_next_instance, next_instance_t)
        return next_instance_t

    def fixed_stop_arrival(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.next_stop[i])
        s = STOPS[curr_stop_idx]
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.next_stop[i] = STOPS[curr_stop_idx + 1]
        drop_offs = self.get_dropoffs()
        self.load[i] -= drop_offs
        bus_load = self.load[i]
        last_bus_time = self.last_bus_time[s]
        assert bus_load >= 0
        if last_bus_time == START_TIME_SEC:
            headway = INIT_HEADWAY
            p_arrivals = self.get_arrivals_start(headway)
            prev_denied = 0
        else:
            headway = self.time - last_bus_time
            if headway < 0:
                headway = 0
            p_arrivals = self.get_pax_arrivals(headway, last_bus_time)
            prev_denied = self.track_denied_boardings[s]
        pax_at_stop = p_arrivals + prev_denied
        allowed = CAPACITY - bus_load
        boardings = min(allowed, pax_at_stop)
        denied = pax_at_stop - boardings
        self.track_denied_boardings[s] = denied
        dwell_time = ACC_DEC_TIME + max(boardings * BOARDING_TIME, drop_offs * ALIGHTING_TIME)
        dwell_time = (boardings + drop_offs > 0) * dwell_time
        self.load[i] += boardings
        self.dep_t[i] = self.time + dwell_time
        self.last_bus_time[s] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
        self.record_trajectories(pickups=boardings, dropoffs=drop_offs, denied_board=denied)
        return

    def terminal_departure(self):
        # first create file for trip
        i = self.bus_idx
        self.create_trip()
        self.last_stop[i] = STOPS[0]
        self.next_stop[i] = STOPS[1]
        s = self.last_stop[i]
        self.dep_t[i] = self.time
        self.terminal_dep_t[i] = self.time
        if self.last_bus_time[s] == START_TIME_SEC:
            headway = INIT_HEADWAY
            boardings = self.get_arrivals_start(headway)
        else:
            headway = self.dep_t[i] - self.last_bus_time[s]
            boardings = self.get_pax_arrivals(headway, self.last_bus_time[s])
        self.load[i] += boardings
        self.record_trajectories(pickups=boardings)
        self.last_bus_time[s] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
        return

    def terminal_arrival(self):
        i = self.bus_idx
        self.arr_t[i] = self.time
        drop_offs = self.load[i]
        self.record_trajectories(dropoffs=drop_offs)
        s = self.next_stop[i]
        if self.last_bus_time[s] == START_TIME_SEC:
            headway = INIT_HEADWAY
        else:
            headway = self.time - self.last_bus_time[s]
            if headway < 0:
                headway = 0
        self.last_bus_time[s] = self.time
        # delete record of trip
        self.remove_trip()
        return

    def next_event(self):
        if self.next_departures:
            next_departure = self.next_departures[0]
            if False not in [next_departure < t for t in self.next_instance_time]:
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
        self.next_trip_ids = ORDERED_TRIPS
        self.time = START_TIME_SEC
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
            self.last_bus_time[s] = self.time
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []
        return False

    def prep(self):
        self.next_event()
        if self.time >= END_TIME_SEC:
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

    def chop_trajectories(self):
        trajectories = dict(self.trajectories)
        for trip in trajectories:
            if trajectories[trip]:
                dep_time = trajectories[trip][0][1]
                if dep_time < FOCUS_START_TIME_SEC or dep_time > FOCUS_END_TIME_SEC:
                    self.trajectories.pop(trip)
            else:
                self.trajectories.pop(trip)
        return

    def process_results(self):
        self.chop_trajectories()
        return


def estimate_arrival_time(start_time, start_stop, end_stop):
    time_control = start_time
    start_stop_idx = STOPS.index(start_stop)
    end_stop_idx = STOPS.index(end_stop)
    for i in range(start_stop_idx, end_stop_idx):
        stop0 = STOPS[i]
        stop1 = STOPS[i+1]
        interv = get_interval(time_control, TIME_INTERVAL_LENGTH_MINS)
        interv_idx = interv - TIME_START_INTERVAL
        mean_runtime = LINK_TIMES_MEAN[stop0+'-'+stop1][interv_idx]
        time_control += mean_runtime
        assert mean_runtime
    arrival_time = time_control
    return arrival_time


class SimulationEnvDeepRL(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(SimulationEnvDeepRL, self).__init__(*args, **kwargs)

        self.trips_sars = {}

    def _add_observations(self):
        t = self.time
        i = self.bus_idx
        stop_id = self.next_stop[i]
        trip_id = self.active_trips[i]
        trip_sars = self.trips_sars[trip_id]

        bus_load = self.load[i]
        forward_headway = t - self.last_bus_time[stop_id]

        # for previous trip
        if i < len(self.active_trips) - 1:
            dep_t = self.dep_t[i+1]
            stop0 = self.last_stop[i+1]
            stop1 = stop_id
        else:
            dep_t = self.next_departures[0]
            stop0 = STOPS[0]
            stop1 = stop_id

        follow_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1)
        backward_headway = follow_trip_arrival_time - t

        new_state = [bus_load, forward_headway, backward_headway]

        if trip_sars:
            prev_sars = trip_sars[-1]
            prev_sars[3] = new_state
            self.trips_sars[trip_id][-1][3] = prev_sars
        new_sars = [new_state, 0, 0, []]
        self.trips_sars[trip_id].append(new_sars)
        return

    def take_action(self):

        return

    def reset_simulation(self):
        self.next_departures = SCHEDULED_DEPARTURES.copy()
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(self.next_departures))]
        self.next_departures = [sum(x) for x in zip(self.next_departures, dep_delays)]
        self.next_trip_ids = [i for i in range(1, 1 + len(SCHEDULED_DEPARTURES))]
        self.time = START_TIME_SEC
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
            self.last_bus_time[s] = self.time
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []

        # RL PARAMETERS
        for trip_id in CONTROLLED_TRIPS:
            self.trips_sars[trip_id] = []

        return False

    def prep(self):
        self.next_event()
        t = self.time
        if t >= END_TIME_SEC:
            return True

        if self.event_type == 2:
            self.terminal_arrival()
            return not self.next_departures and not self.next_instance_time

        if self.event_type == 1:
            i = self.bus_idx
            arrival_stop = self.next_stop[i]
            trip_id = self.active_trips[i]
            if arrival_stop in CONTROLLED_STOPS and trip_id in CONTROLLED_TRIPS:

                return False
            self.fixed_stop_arrival()
            return self.prep()

        if self.event_type == 0:
            self.terminal_departure()
            return self.prep()