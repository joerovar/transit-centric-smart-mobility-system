from input import *
import numpy as np
from scipy.stats import lognorm
import random


def get_interval(t, interval_length):
    interval = int(t / (interval_length * 60))
    return interval


class SimulationEnv:
    def __init__(self):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        # SIMUL PARAMS
        # NEW
        self.no_overtake_policy = True
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
        self.stop_wait_time = {}
        self.denied_boardings = {}
        self.tot_pax_at_stop = {}
        self.adjusted_wait_time = {}
        self.wait_time_from_h = {}
        self.tot_denied_boardings = {}

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
        e_pax = arrival_rates[inter - DEM_START_INTERVAL] * headway/60
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
            e_pax += arrival_rates[inter - DEM_START_INTERVAL] * ((min(edge, t2) - t)/60)
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
            prev_bus_idx = curr_bus_idx-1
            try:
                prev_bus_last_stop = self.last_stop[prev_bus_idx]
            except IndexError:
                print(curr_bus_idx)
                print(prev_bus_idx)
                print(self.last_stop)
                raise
            curr_bus_next_stop = self.next_stop[curr_bus_idx]
            if prev_bus_last_stop == curr_bus_next_stop:
                prev_bus_next_instance = self.next_instance_time[prev_bus_idx]
                next_instance_t = prev_bus_next_instance
        return next_instance_t

    def fixed_stop_arrival(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.next_stop[i])
        s = STOPS[curr_stop_idx]
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.next_stop[i] = STOPS[curr_stop_idx + 1]
        drop_offs = self.get_dropoffs()
        self.load[i] -= drop_offs
        load = self.load[i]
        last_bus_time = self.last_bus_time[s]
        assert load >= 0
        if last_bus_time == START_TIME_SEC:
            headway = INIT_HEADWAY
            p_arrivals = self.get_arrivals_start(headway)
        else:
            headway = self.time - last_bus_time
            if headway < 0:
                headway = 0
            self.recorded_headway[s].append(headway)
            p_arrivals = self.get_pax_arrivals(headway, last_bus_time)

        self.tot_pax_at_stop[s] += p_arrivals
        prev_denied = self.denied_boardings[s]
        pax_at_stop = p_arrivals + prev_denied
        allowed = CAPACITY - load
        boardings = min(allowed, pax_at_stop)
        denied = pax_at_stop - boardings
        self.tot_denied_boardings[s] += denied
        self.denied_boardings[s] = denied
        self.stop_wait_time[s] += (prev_denied + p_arrivals/2) * headway
        dwell_time = ACC_DEC_TIME + boardings * BOARDING_TIME + drop_offs * ALIGHTING_TIME
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
            self.recorded_headway[s].append(headway)
        actual_route_time = self.time - self.terminal_dep_t[i]
        self.recorded_trip_times.append(actual_route_time)
        self.last_bus_time[s] = self.time
        # delete record of trip
        self.remove_trip()
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
            self.recorded_headway[self.last_stop[i]].append(headway)
            boardings = self.get_pax_arrivals(headway, self.last_bus_time[s])
        self.tot_pax_at_stop[s] += boardings
        self.stop_wait_time[s] += boardings * headway/2
        self.load[i] += boardings
        self.record_trajectories(pickups=boardings)
        self.last_bus_time[s] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
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
            self.recorded_headway[s] = []
            self.last_bus_time[s] = self.time
            self.stop_wait_time[s] = 0
            self.denied_boardings[s] = 0
            self.tot_pax_at_stop[s] = 0
            self.tot_denied_boardings[s] = 0

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []
        self.recorded_trip_times = []
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
        trajectories = self.trajectories
        for trip in trajectories:
            trajectory = trajectories[trip]
            start_idx = 0
            end_idx = -1
            found_start = False
            for i in range(len(trajectory)):
                if trajectory[i][1] >= FOCUS_START_TIME:
                    if not found_start:
                        found_start = True
                        start_idx = i
                if trajectory[i][1] > FOCUS_END_TIME:
                    end_idx = i - 1
                    break
            trajectories[trip] = trajectory[start_idx:end_idx]
        return

    def process_results(self):
        for s in self.stop_wait_time.keys():
            if self.tot_pax_at_stop[s]:
                self.adjusted_wait_time[s] = self.stop_wait_time[s] / self.tot_pax_at_stop[s]
            else:
                self.adjusted_wait_time[s] = 0
            headway = np.array(self.recorded_headway[s])
            mean_headway = headway.mean()
            cv_headway = headway.std() / mean_headway
            self.wait_time_from_h[s] = (mean_headway / 2) * (1 + (cv_headway * cv_headway))
        self.chop_trajectories()
        return



