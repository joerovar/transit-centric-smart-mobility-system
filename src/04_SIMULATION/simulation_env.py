from input import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from classes_simul import Passenger, Stop, Trip


def get_interval(t, interval_length):
    # time in seconds, interval length in minutes
    interval = int(t / (interval_length * 60))
    return interval


def estimate_arrival_time(start_time, start_stop, end_stop, time_dependent_tt):
    time_control = start_time
    start_stop_idx = STOPS.index(start_stop)
    end_stop_idx = STOPS.index(end_stop)
    for i in range(start_stop_idx, end_stop_idx):
        stop0 = STOPS[i]
        stop1 = STOPS[i + 1]
        if time_dependent_tt:
            interv = get_interval(time_control, TIME_INTERVAL_LENGTH_MINS)
            interv_idx = interv - TIME_START_INTERVAL
            mean_runtime = LINK_TIMES_MEAN[stop0 + '-' + stop1][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[stop0 + '-' + stop1]
        time_control += mean_runtime
        assert mean_runtime
    arrival_time = time_control
    return arrival_time


def _compute_reward(action, fw_h, bw_h, trip_id, prev_bw_h, prev_fw_h, prev_pax_at_s):
    trip_idx = ORDERED_TRIPS.index(trip_id)
    # follow_trip_id = ORDERED_TRIPS[trip_idx + 1]
    # lead_trip_id = ORDERED_TRIPS[trip_idx - 1]

    hw_diff0 = abs(prev_fw_h - prev_bw_h)
    hw_diff1 = abs(fw_h - bw_h)
    reward = hw_diff0 - hw_diff1
    if prev_pax_at_s and action == SKIP_ACTION:
        reward -= 0.5*prev_bw_h
    return reward


class SimulationEnv:
    def __init__(self, no_overtake_policy=True, time_dependent_travel_time=True, time_dependent_demand=True):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        self.no_overtake_policy = no_overtake_policy
        self.time_dependent_travel_time = time_dependent_travel_time
        self.time_dependent_demand = time_dependent_demand
        self.next_actual_departures = []
        self.next_trip_ids = []
        self.active_trips = []
        self.last_stop = []
        self.next_stop = []
        self.dep_t = []
        self.arr_t = []
        self.offs = []
        self.ons = []
        self.pax_at_stop = []
        self.denied = []
        self.load = []
        self.bus_idx = 0
        self.event_type = 0
        self.next_instance_time = []
        self.time = 0.0
        self.last_bus_time = {}
        self.track_denied_boardings = {}
        # RECORDINGS
        self.trajectories = {}

    def record_trajectories(self, pickups=0, offs=0, denied_board=0, hold=0, skip=False):
        i = self.bus_idx
        trip_id = self.active_trips[i]
        trajectory = [self.last_stop[i], round(self.arr_t[i], 1), round(self.dep_t[i], 1),
                      self.load[i], pickups, offs, denied_board, hold, int(skip)]
        self.trajectories[trip_id].append(trajectory)
        return

    def create_trip(self):
        trip_file = [
            self.next_instance_time, self.last_stop, self.next_stop, self.load,
            self.dep_t, self.arr_t, self.ons, self.offs, self.denied, self.pax_at_stop
        ]
        for tf in trip_file:
            tf.append(0)
        return

    def remove_trip(self):
        trip_file = [
            self.active_trips, self.next_instance_time, self.last_stop, self.next_stop, self.load,
            self.dep_t, self.arr_t, self.ons, self.offs, self.denied, self.pax_at_stop
        ]
        for tf in trip_file:
            tf.pop(self.bus_idx)
        return

    def get_arrivals_start(self, headway):
        i = self.bus_idx
        stop = self.last_stop[i]
        if self.time_dependent_demand:
            arrival_rates = ARRIVAL_RATES[stop]
            inter = get_interval(self.time, DEM_INTERVAL_LENGTH_MINS)
            e_pax = (arrival_rates[inter - DEM_START_INTERVAL] / 60) * headway / 60
        else:
            arrival_rate = ARRIVAL_RATE[stop]
            e_pax = (arrival_rate / 60) * headway / 60
        pax_at_stop = np.random.poisson(lam=e_pax)
        return pax_at_stop

    def get_pax_arrivals(self, headway, last_arrival_t):
        i = self.bus_idx
        stop = self.last_stop[i]
        if self.time_dependent_demand:
            arrival_rates = ARRIVAL_RATES[stop]
            e_pax = 0
            t = last_arrival_t
            t2 = last_arrival_t + headway
            while t < t2:
                inter = get_interval(t, DEM_INTERVAL_LENGTH_MINS)
                edge = (inter + 1) * 60 * DEM_INTERVAL_LENGTH_MINS
                e_pax += (arrival_rates[inter - DEM_START_INTERVAL] / 60) * ((min(edge, t2) - t) / 60)
                t = min(edge, t2)
        else:
            arrival_rate = ARRIVAL_RATE[stop]
            e_pax = (arrival_rate / 60) * headway / 60
        pax_arrivals = np.random.poisson(lam=e_pax)
        return pax_arrivals

    def get_offs(self):
        i = self.bus_idx
        stop = self.last_stop[i]
        if self.time_dependent_demand:
            interv = get_interval(self.time, DEM_INTERVAL_LENGTH_MINS)
            interv_idx = interv - DEM_START_INTERVAL
            offs = int(round(ALIGHT_FRACTIONS[stop][interv_idx] * self.load[i]))
        else:
            offs = int(round(ALIGHT_FRACTION[stop] * self.load[i]))
        return offs

    def get_travel_time(self):
        i = self.bus_idx
        prev_stop = self.last_stop[i]
        next_stop = self.next_stop[i]
        link = str(prev_stop) + '-' + str(next_stop)
        if self.time_dependent_travel_time:
            interv = get_interval(self.time, TIME_INTERVAL_LENGTH_MINS)
            interv_idx = interv - TIME_START_INTERVAL
            mean_runtime = LINK_TIMES_MEAN[link][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[link]
        assert mean_runtime
        s = LOGN_S
        runtime = lognorm.rvs(s, scale=mean_runtime)
        assert TTD == 'LOGNORMAL'
        return runtime

    def no_overtake(self):
        # if this is the first bus then ignore and return
        # i is subject bus index and j = i - 1 (leading bus)
        i = self.bus_idx
        next_instance_t = self.next_instance_time[i]
        if i == -1:
            i = self.active_trips.index(self.active_trips[i])
        if i:
            lead_bus_idx = i - 1
            try:
                lead_bus_next_stop = self.next_stop[lead_bus_idx]
            except IndexError:
                print(i, lead_bus_idx, self.last_stop[i])
                raise
            curr_bus_next_stop = self.next_stop[i]
            if lead_bus_next_stop == curr_bus_next_stop:
                lead_bus_next_instance = self.next_instance_time[lead_bus_idx]
                next_instance_t = max(lead_bus_next_instance, next_instance_t)
        return next_instance_t

    def fixed_stop_unload(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.next_stop[i])
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.next_stop[i] = STOPS[curr_stop_idx + 1]
        self.arr_t[i] = self.time

        self.offs[i] = self.get_offs()
        self.load[i] -= self.offs[i]
        return

    def fixed_stop_arrivals(self):
        i = self.bus_idx
        bus_load = self.load[i]
        s = self.last_stop[i]
        last_bus_time = self.last_bus_time[s]
        assert bus_load >= 0
        if last_bus_time:
            headway = self.time - last_bus_time
            if headway < 0:
                headway = 0
            p_arrivals = self.get_pax_arrivals(headway, last_bus_time)
            prev_denied = self.track_denied_boardings[s]
        else:
            headway = INIT_HEADWAY
            p_arrivals = self.get_arrivals_start(headway)
            prev_denied = 0
        self.pax_at_stop[i] = p_arrivals + prev_denied
        return

    def fixed_stop_load(self):
        i = self.bus_idx
        bus_load = self.load[i]
        s = self.last_stop[i]
        pax_at_stop = self.pax_at_stop[i]
        allowed = CAPACITY - bus_load
        self.ons[i] = min(allowed, pax_at_stop)
        self.denied[i] = pax_at_stop - self.ons[i]
        self.track_denied_boardings[s] = self.denied[i]
        self.load[i] += self.ons[i]
        return

    def fixed_stop_depart(self):
        i = self.bus_idx
        ons = self.ons[i]
        offs = self.offs[i]
        denied = self.denied[i]
        s = self.last_stop[i]

        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = ACC_DEC_TIME + max(ons * BOARDING_TIME, offs * ALIGHTING_TIME) + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (ons + offs > 0) * dwell_time

        self.dep_t[i] = self.time + dwell_time

        if self.no_overtake_policy and self.last_bus_time[s]:
            self.dep_t[i] = max(self.last_bus_time[s], self.dep_t[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])

        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
        self.record_trajectories(pickups=ons, offs=offs, denied_board=denied)
        return

    def terminal_departure(self):
        # first create file for trip
        i = self.bus_idx
        self.create_trip()
        self.last_stop[i] = STOPS[0]
        self.next_stop[i] = STOPS[1]
        s = self.last_stop[i]
        self.arr_t[i] = self.time
        self.dep_t[i] = self.time
        last_bus_time = self.last_bus_time[s]
        if last_bus_time:
            headway = self.dep_t[i] - last_bus_time
            boardings = self.get_pax_arrivals(headway, last_bus_time)
        else:
            headway = INIT_HEADWAY
            boardings = self.get_arrivals_start(headway)
        self.load[i] += boardings
        self.record_trajectories(pickups=boardings)
        self.last_bus_time[s] = deepcopy(self.dep_t[i])
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
        return

    def terminal_arrival(self):
        i = self.bus_idx
        self.arr_t[i] = self.time
        offs = self.load[i]
        self.load[i] = 0
        self.last_stop[i] = self.next_stop[i]
        self.dep_t[i] = self.arr_t[i]
        self.record_trajectories(offs=offs)
        # delete record of trip
        self.remove_trip()
        return

    def next_event(self):
        if self.next_actual_departures:
            next_departure = self.next_actual_departures[0]
            if False not in [next_departure < t for t in self.next_instance_time]:
                self.time = self.next_actual_departures[0]
                self.active_trips.append(self.next_trip_ids[0])
                self.bus_idx = -1
                self.event_type = 0
                self.next_actual_departures.pop(0)
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
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(SCHEDULED_DEPARTURES))]
        self.next_actual_departures = [sum(x) for x in zip(SCHEDULED_DEPARTURES, dep_delays)]
        self.next_trip_ids = deepcopy(ORDERED_TRIPS)
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
        self.offs = []
        self.ons = []
        self.denied = []
        self.pax_at_stop = []
        self.event_type = 0
        # stop-level data
        for s in STOPS:
            self.last_bus_time[s] = []
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []
        return False

    def prep(self):
        self.next_event()
        if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
            return True
        if self.event_type == 2:
            self.terminal_arrival()
            return False

        if self.event_type == 1:
            self.fixed_stop_unload()
            self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.event_type == 0:
            self.terminal_departure()
            return False

    def chop_trajectories(self):
        for trip in self.trajectories.copy():
            if trip not in FOCUS_TRIPS:
                self.trajectories.pop(trip)
        return

    def process_results(self):
        self.chop_trajectories()
        return


class SimulationEnvWithControl(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(SimulationEnvWithControl, self).__init__(*args, **kwargs)

    def fixed_stop_depart(self, hold=0):
        i = self.bus_idx
        ons = self.ons[i]
        offs = self.offs[i]
        denied = self.denied[i]
        s = self.last_stop[i]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if hold:
            dwell_time_pax = max(ACC_DEC_TIME + ons * BOARDING_TIME, ACC_DEC_TIME + offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (ons + offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(ons * BOARDING_TIME, offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (ons + offs > 0) * dwell_time

        self.dep_t[i] = self.time + dwell_time

        if self.no_overtake_policy and self.last_bus_time[s]:
            self.dep_t[i] = max(self.last_bus_time[s], self.dep_t[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()

        self.record_trajectories(pickups=ons, offs=offs, denied_board=denied, hold=hold)
        return

    def decide_bus_holding(self):
        t = self.time
        i = self.bus_idx
        stop_id = self.last_stop[i]
        trip_id = self.active_trips[i]

        forward_headway = t - self.last_bus_time[stop_id]

        # for previous trip
        if i < len(self.active_trips) - 1:
            dep_t = self.dep_t[i + 1]
            stop0 = self.last_stop[i + 1]
            stop1 = stop_id
        else:
            # in case there is no trip before we can look at future departures which always exist
            # we look at scheduled departures and not actual which include distributed delays
            trip_idx = ORDERED_TRIPS.index(trip_id) + 1
            dep_t = SCHEDULED_DEPARTURES[trip_idx]
            stop0 = STOPS[0]
            stop1 = stop_id

        follow_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
        backward_headway = follow_trip_arrival_time - t

        if backward_headway < 0:
            backward_headway = 0

        self.fixed_stop_arrivals()
        self.fixed_stop_load()

        if backward_headway > forward_headway:
            holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
            self.fixed_stop_depart(hold=holding_time)
        else:
            self.fixed_stop_depart()
        return

    def prep(self):
        self.next_event()

        if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
            return True

        if self.event_type == 2:
            self.terminal_arrival()
            return False

        if self.event_type == 1:
            i = self.bus_idx
            arrival_stop = self.next_stop[i]
            trip_id = self.active_trips[i]
            if (trip_id != ORDERED_TRIPS[0]) & (trip_id != ORDERED_TRIPS[-1]):
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.fixed_stop_unload()
                    self.decide_bus_holding()
                    return False
            self.fixed_stop_unload()
            self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.event_type == 0:
            self.terminal_departure()
            return False


class SimulationEnvDeepRL(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(SimulationEnvDeepRL, self).__init__(*args, **kwargs)
        self.trips_sars = {}
        self.bool_terminal_state = False

    def reset_simulation(self):
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(SCHEDULED_DEPARTURES))]
        self.next_actual_departures = [sum(x) for x in zip(SCHEDULED_DEPARTURES, dep_delays)]
        self.next_trip_ids = deepcopy(ORDERED_TRIPS)
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
        self.offs = []
        self.ons = []
        self.denied = []
        self.pax_at_stop = []
        self.event_type = 0

        # stop-level data
        for s in STOPS:
            self.last_bus_time[s] = []
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}

        # RL PARAMETERS
        for trip_id in self.next_trip_ids:
            self.trajectories[trip_id] = []
            self.trips_sars[trip_id] = []
        self.bool_terminal_state = False
        return False

    def _add_observations(self):
        t = self.time
        i = self.bus_idx
        stop_id = self.last_stop[i]
        stop_idx = STOPS.index(self.last_stop[i])
        trip_id = self.active_trips[i]
        trip_sars = self.trips_sars[trip_id]

        bus_load = self.load[i]
        forward_headway = t - self.last_bus_time[stop_id]

        # for previous trip
        if i < len(self.active_trips) - 1:
            dep_t = self.dep_t[i + 1]
            stop0 = self.last_stop[i + 1]
            stop1 = stop_id
        else:
            # in case there is no trip before we can look at future departures which always exist
            # we look at scheduled departures and not actual which include distributed delays
            trip_idx = ORDERED_TRIPS.index(trip_id) + 1
            dep_t = SCHEDULED_DEPARTURES[trip_idx]
            stop0 = STOPS[0]
            stop1 = stop_id

        follow_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
        backward_headway = follow_trip_arrival_time - t
        if backward_headway < 0:
            backward_headway = 0

        route_progress = stop_idx/len(STOPS)
        new_state = [route_progress, bus_load, forward_headway, backward_headway, self.pax_at_stop[i]]

        if trip_sars:
            prev_sars = self.trips_sars[trip_id][-1]
            previous_action = prev_sars[1]
            previous_backward_headway = prev_sars[0][IDX_BW_H]
            prev_fw_h = prev_sars[0][IDX_FW_H]
            prev_pax_at_stop = prev_sars[0][IDX_PAX_AT_STOP]
            self.trips_sars[trip_id][-1][2] = _compute_reward(previous_action, forward_headway, backward_headway,
                                                              trip_id, previous_backward_headway, prev_fw_h,
                                                              prev_pax_at_stop)
            self.trips_sars[trip_id][-1][3] = new_state
        if not self.bool_terminal_state:
            new_sars = [new_state, 0, 0, []]
            self.trips_sars[trip_id].append(new_sars)
        return

    def fixed_stop_arrivals(self):
        i = self.bus_idx
        bus_load = self.load[i]
        s = self.last_stop[i]
        last_bus_time = self.last_bus_time[s]
        assert bus_load >= 0
        if last_bus_time:
            headway = self.time - last_bus_time
            if headway < 0:
                headway = 0
            p_arrivals = self.get_pax_arrivals(headway, last_bus_time)
            prev_denied = self.track_denied_boardings[s]
        else:
            headway = INIT_HEADWAY
            p_arrivals = self.get_arrivals_start(headway)
            prev_denied = 0
        self.pax_at_stop[i] = p_arrivals + prev_denied
        return

    def fixed_stop_load(self, skip=False):
        i = self.bus_idx
        bus_load = self.load[i]
        pax_at_stop = self.pax_at_stop[i]
        s = self.last_stop[i]
        ons = self.ons[i]
        if skip:
            self.ons[i] = 0
            self.denied[i] = pax_at_stop
        else:
            allowed = CAPACITY - bus_load
            self.ons[i] = min(allowed, pax_at_stop)
            self.denied[i] = pax_at_stop - self.ons[i]
        self.track_denied_boardings[s] = self.denied[i]
        self.load[i] += ons
        return

    def take_action(self, action):
        i = self.bus_idx
        # record action in sars
        trip_id = self.active_trips[i]
        self.trips_sars[trip_id][-1][1] = action

        if action:
            self.fixed_stop_load()
            self.fixed_stop_depart(hold=(action - 1) * BASE_HOLDING_TIME)
        else:
            self.fixed_stop_load(skip=True)
            self.fixed_stop_depart(skip=True)
        return

    def fixed_stop_depart(self, hold=0, skip=False):
        i = self.bus_idx
        ons = self.ons[i]
        offs = self.offs[i]
        denied = self.denied[i]
        s = self.last_stop[i]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if skip:
            dwell_time = ACC_DEC_TIME + ALIGHTING_TIME * offs + dwell_time_error
            dwell_time = (offs > 0) * dwell_time
        elif hold:
            dwell_time_pax = max(ACC_DEC_TIME + ons * BOARDING_TIME, ACC_DEC_TIME + offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (ons + offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(ons * BOARDING_TIME, offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (ons + offs > 0) * dwell_time

        self.dep_t[i] = self.time + dwell_time

        if self.no_overtake_policy and self.last_bus_time[s]:
            self.dep_t[i] = max(self.last_bus_time[s], self.dep_t[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()

        self.record_trajectories(pickups=ons, offs=offs, denied_board=denied, hold=hold, skip=skip)
        return

    def prep(self):
        self.next_event()
        if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
            return True

        if self.event_type == 2:
            self.terminal_arrival()
            return self.prep()

        if self.event_type == 1:
            i = self.bus_idx
            arrival_stop = self.next_stop[i]
            trip_id = self.active_trips[i]
            if (trip_id != ORDERED_TRIPS[0]) & (trip_id != ORDERED_TRIPS[-1]):
                if arrival_stop in CONTROLLED_STOPS:
                    if arrival_stop == CONTROLLED_STOPS[-1]:
                        self.bool_terminal_state = True
                        self.fixed_stop_unload()
                        self._add_observations()
                        self.fixed_stop_arrivals()
                        self.fixed_stop_load()
                        self.fixed_stop_depart()
                    else:
                        self.bool_terminal_state = False
                        self.fixed_stop_unload()
                        self.fixed_stop_arrivals()
                        self._add_observations()
                    return False
            self.fixed_stop_unload()
            self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return self.prep()

        if self.event_type == 0:
            self.terminal_departure()
            return self.prep()


class DetailedSimulationEnv(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnv, self).__init__(*args, **kwargs)
        self.stops = []
        self.trips = []
        self.completed_pax = []

    def reset_simulation(self):
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(SCHEDULED_DEPARTURES))]
        self.next_actual_departures = [sum(x) for x in zip(SCHEDULED_DEPARTURES, dep_delays)]
        self.next_trip_ids = deepcopy(ORDERED_TRIPS)
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
        self.offs = []
        self.ons = []
        self.denied = []
        self.pax_at_stop = []
        self.event_type = 0

        # stop-level data
        for s in STOPS:
            self.last_bus_time[s] = []
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}
        for i in self.next_trip_ids:
            self.trajectories[i] = []

        # initialize vehicle trips
        self.trips = []
        for i in range(len(ORDERED_TRIPS)):
            self.trips.append(Trip(ORDERED_TRIPS[i]))

        # initialize passenger demand
        self.completed_pax = []
        self.initialize_pax_demand()
        return False

    def initialize_pax_demand(self):
        pax_info = {}
        self.stops = []
        for i in range(len(STOPS)):
            self.stops.append(Stop(STOPS[i]))
            pax_info['arr_times'] = []
            pax_info['o_stop_idx'] = []
            pax_info['d_stop_idx'] = []
            for j in range(i + 1, len(STOPS)):
                for k in range(DEM_START_INTERVAL, DEM_END_INTERVAL):
                    start_edge_interval = k * DEM_INTERVAL_LENGTH_MINS * 60
                    end_edge_interval = start_edge_interval + DEM_INTERVAL_LENGTH_MINS * 60
                    od_rate = ODT[k - DEM_START_INTERVAL, i, j]
                    if not np.isnan(od_rate):
                        max_size = int(od_rate * (DEM_INTERVAL_LENGTH_MINS / 60) * 3)
                        if od_rate > 0:
                            temp_pax_interarr_times = np.random.exponential(3600 / od_rate, size=max_size)
                            temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
                            if k == DEM_START_INTERVAL:
                                temp_pax_arr_times += PAX_INIT_TIME[i]
                            else:
                                temp_pax_arr_times += max(start_edge_interval, PAX_INIT_TIME[i])
                            temp_pax_arr_times = temp_pax_arr_times[
                                temp_pax_arr_times <= min(END_TIME_SEC, end_edge_interval)]
                            temp_pax_arr_times = temp_pax_arr_times.tolist()
                            if len(temp_pax_arr_times):
                                pax_info['arr_times'] += temp_pax_arr_times
                                pax_info['o_stop_idx'] += [i] * len(temp_pax_arr_times)
                                pax_info['d_stop_idx'] += [j] * len(temp_pax_arr_times)
            df = pd.DataFrame(pax_info).sort_values(by='arr_times')
            pax_sorted_info = df.to_dict('list')
            for o, d, at in zip(pax_sorted_info['o_stop_idx'], pax_sorted_info['d_stop_idx'],
                                pax_sorted_info['arr_times']):
                self.stops[i].pax.append(Passenger(o, d, at))
        return

    def fixed_stop_unload(self):
        # SWITCH POSITION AND ARRIVAL TIME
        i = self.bus_idx
        curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        curr_stop_idx = STOPS.index(self.next_stop[i])
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.next_stop[i] = STOPS[curr_stop_idx + 1]
        self.arr_t[i] = self.time
        self.offs[i] = 0
        for p in self.trips[curr_trip_idx].pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.trips[curr_trip_idx].pax.remove(p)
                self.offs[i] += 1
        self.load[i] -= self.offs[i]
        return

    def fixed_stop_arrivals(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.last_stop[i])
        self.pax_at_stop[i] = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                self.pax_at_stop[i] += 1
            else:
                break
        return

    def fixed_stop_load(self):
        i = self.bus_idx
        bus_load = self.load[i]
        curr_stop_idx = STOPS.index(self.last_stop[i])
        curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        self.denied[i] = 0
        assert bus_load >= 0
        self.ons[i] = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + self.ons[i] + 1 <= CAPACITY:
                    p.trip_id = self.active_trips[i]
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    self.trips[curr_trip_idx].pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    self.ons[i] += 1
                else:
                    p.denied = 1
                    self.denied[i] += 1
            else:
                break
        self.load[i] += self.ons[i]
        return

    def terminal_departure(self):
        # first create file for trip
        i = self.bus_idx
        self.create_trip()
        self.last_stop[i] = STOPS[0]
        self.next_stop[i] = STOPS[1]
        s = self.last_stop[i]
        self.arr_t[i] = self.time
        self.dep_t[i] = self.time
        curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        bus_load = self.load[i]
        self.denied[i] = 0
        for p in self.stops[0].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + self.ons[i] + 1 <= CAPACITY:
                    p.trip_id = self.active_trips[i]
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    self.trips[curr_trip_idx].pax.append(p)
                    self.stops[0].pax.remove(p)
                    self.ons[i] += 1
                else:
                    p.denied = 1
                    self.denied[i] += 1
        self.load[i] += self.ons[i]
        self.record_trajectories(pickups=self.ons[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])

        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()
        return

    def terminal_arrival(self):
        i = self.bus_idx
        self.arr_t[i] = self.time
        curr_stop_idx = STOPS.index(self.next_stop[i])
        curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        for p in self.trips[curr_trip_idx].pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.trips[curr_trip_idx].pax.remove(p)
                self.offs[i] += 1
        self.load[i] = 0
        self.last_stop[i] = STOPS[curr_stop_idx]
        self.dep_t[i] = self.arr_t[i]
        self.record_trajectories(offs=self.offs[i])
        # delete record of trip
        self.remove_trip()
        return

    def chop_pax(self):
        for p in self.completed_pax.copy():
            if p.trip_id not in FOCUS_TRIPS:
                self.completed_pax.remove(p)
        return

    def process_results(self):
        self.chop_trajectories()
        self.chop_pax()
        return


class DetailedSimulationEnvWithControl(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithControl, self).__init__(*args, **kwargs)

    def fixed_stop_depart(self, hold=0):
        i = self.bus_idx
        ons = self.ons[i]
        offs = self.offs[i]
        denied = self.denied[i]
        s = self.last_stop[i]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if hold:
            dwell_time_pax = max(ACC_DEC_TIME + ons * BOARDING_TIME, ACC_DEC_TIME + offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (ons + offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(ons * BOARDING_TIME, offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (ons + offs > 0) * dwell_time

        self.dep_t[i] = self.time + dwell_time

        if self.no_overtake_policy and self.last_bus_time[s]:
            self.dep_t[i] = max(self.last_bus_time[s], self.dep_t[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()

        self.record_trajectories(pickups=ons, offs=offs, denied_board=denied, hold=hold)
        return

    def decide_bus_holding(self):
        t = self.time
        i = self.bus_idx
        stop_id = self.last_stop[i]
        trip_id = self.active_trips[i]

        forward_headway = t - self.last_bus_time[stop_id]

        # for previous trip
        if i < len(self.active_trips) - 1:
            dep_t = self.dep_t[i + 1]
            stop0 = self.last_stop[i + 1]
            stop1 = stop_id
        else:
            # in case there is no trip before we can look at future departures which always exist
            # we look at scheduled departures and not actual which include distributed delays
            trip_idx = ORDERED_TRIPS.index(trip_id) + 1
            dep_t = SCHEDULED_DEPARTURES[trip_idx]
            stop0 = STOPS[0]
            stop1 = stop_id

        follow_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
        backward_headway = follow_trip_arrival_time - t
        if backward_headway < 0:
            backward_headway = 0
        self.fixed_stop_arrivals()
        self.fixed_stop_load()
        if backward_headway > forward_headway:
            holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
            self.fixed_stop_depart(hold=holding_time)
        else:
            self.fixed_stop_depart()
        return

    def prep(self):
        self.next_event()

        if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
            return True

        if self.event_type == 2:
            self.terminal_arrival()
            return False

        if self.event_type == 1:
            i = self.bus_idx
            arrival_stop = self.next_stop[i]
            trip_id = self.active_trips[i]
            if (trip_id != ORDERED_TRIPS[0]) & (trip_id != ORDERED_TRIPS[-1]):
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.fixed_stop_unload()
                    self.decide_bus_holding()
                    return False
            self.fixed_stop_unload()
            self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.event_type == 0:
            self.terminal_departure()
            return False


class DetailedSimulationEnvWithDeepRL(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithDeepRL, self).__init__(*args, **kwargs)
        self.trips_sars = {}
        self.bool_terminal_state = False

    def reset_simulation(self):
        dep_delays = [max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0) for i in range(len(SCHEDULED_DEPARTURES))]
        self.next_actual_departures = [sum(x) for x in zip(SCHEDULED_DEPARTURES, dep_delays)]
        self.next_trip_ids = deepcopy(ORDERED_TRIPS)
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
        self.offs = []
        self.ons = []
        self.denied = []
        self.pax_at_stop = []
        self.event_type = 0

        # stop-level data
        for s in STOPS:
            self.last_bus_time[s] = []
            self.track_denied_boardings[s] = 0

        # for records
        self.trajectories = {}

        # initialize vehicle trips
        self.trips = []
        for i in range(len(ORDERED_TRIPS)):
            self.trips.append(Trip(ORDERED_TRIPS[i]))

        # Initialize pax demand
        self.completed_pax = []
        self.initialize_pax_demand()

        # RL PARAMETERS
        for trip_id in self.next_trip_ids:
            self.trajectories[trip_id] = []
            self.trips_sars[trip_id] = []
        self.bool_terminal_state = False
        return False

    def fixed_stop_arrivals(self):
        i = self.bus_idx
        curr_stop_idx = STOPS.index(self.last_stop[i])
        self.pax_at_stop[i] = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                self.pax_at_stop[i] += 1
            else:
                break
        return

    def fixed_stop_load(self, skip=False):
        i = self.bus_idx
        bus_load = self.load[i]
        s = self.last_stop[i]
        curr_stop_idx = STOPS.index(self.last_stop[i])
        curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        assert bus_load >= 0
        self.denied[i] = 0
        self.ons[i] = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + self.ons[i] + 1 <= CAPACITY and not skip:
                    p.trip_id = self.active_trips[i]
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    self.trips[curr_trip_idx].pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    self.ons[i] += 1
                else:
                    p.denied = 1
                    self.denied[i] += 1
            else:
                break
        self.track_denied_boardings[s] = self.denied[i]
        self.load[i] += self.ons[i]
        return

    def take_action(self, action):
        i = self.bus_idx
        # record action in sars
        trip_id = self.active_trips[i]
        self.trips_sars[trip_id][-1][1] = action

        if action:
            self.fixed_stop_load()
            self.fixed_stop_depart(hold=(action - 1) * BASE_HOLDING_TIME)
        else:
            self.fixed_stop_load(skip=True)
            self.fixed_stop_depart(skip=True)
        return

    def fixed_stop_depart(self, hold=0, skip=False):
        i = self.bus_idx
        ons = self.ons[i]
        offs = self.offs[i]
        denied = self.denied[i]
        s = self.last_stop[i]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if skip:
            dwell_time = ACC_DEC_TIME + ALIGHTING_TIME * offs + dwell_time_error
            dwell_time = (offs > 0) * dwell_time
        elif hold:
            dwell_time_pax = max(ACC_DEC_TIME + ons * BOARDING_TIME, ACC_DEC_TIME + offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (ons + offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(ons * BOARDING_TIME, offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (ons + offs > 0) * dwell_time

        self.dep_t[i] = self.time + dwell_time

        if self.no_overtake_policy and self.last_bus_time[s]:
            self.dep_t[i] = max(self.last_bus_time[s], self.dep_t[i])
        self.last_bus_time[s] = deepcopy(self.dep_t[i])
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        if self.no_overtake_policy:
            self.next_instance_time[i] = self.no_overtake()

        self.record_trajectories(pickups=ons, offs=offs, denied_board=denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        t = self.time
        i = self.bus_idx
        stop_id = self.last_stop[i]
        stop_idx = STOPS.index(self.last_stop[i])
        trip_id = self.active_trips[i]
        trip_sars = self.trips_sars[trip_id]

        bus_load = self.load[i]
        forward_headway = t - self.last_bus_time[stop_id]

        # for previous trip
        if i < len(self.active_trips) - 1:
            dep_t = self.dep_t[i + 1]
            stop0 = self.last_stop[i + 1]
            stop1 = stop_id
        else:
            # in case there is no trip before we can look at future departures which always exist
            # we look at scheduled departures and not actual which include distributed delays
            trip_idx = ORDERED_TRIPS.index(trip_id) + 1
            dep_t = SCHEDULED_DEPARTURES[trip_idx]
            stop0 = STOPS[0]
            stop1 = stop_id

        follow_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
        backward_headway = follow_trip_arrival_time - t
        if backward_headway < 0:
            backward_headway = 0

        route_progress = stop_idx/len(STOPS)
        new_state = [route_progress, bus_load, forward_headway, backward_headway, self.pax_at_stop[i]]

        if trip_sars:
            prev_sars = self.trips_sars[trip_id][-1]
            previous_action = prev_sars[1]
            previous_backward_headway = prev_sars[0][IDX_BW_H]
            prev_fw_h = prev_sars[0][IDX_FW_H]
            prev_pax_at_stop = prev_sars[0][IDX_PAX_AT_STOP]
            self.trips_sars[trip_id][-1][2] = _compute_reward(previous_action, forward_headway, backward_headway,
                                                              trip_id, previous_backward_headway, prev_fw_h,
                                                              prev_pax_at_stop)
            self.trips_sars[trip_id][-1][3] = new_state
        if not self.bool_terminal_state:
            new_sars = [new_state, 0, 0, []]
            self.trips_sars[trip_id].append(new_sars)
        return

    def prep(self):
        self.next_event()

        if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
            return True

        if self.event_type == 2:
            self.terminal_arrival()
            return self.prep()

        if self.event_type == 1:
            i = self.bus_idx
            arrival_stop = self.next_stop[i]
            trip_id = self.active_trips[i]
            if (trip_id != ORDERED_TRIPS[0]) & (trip_id != ORDERED_TRIPS[-1]):
                if arrival_stop in CONTROLLED_STOPS:
                    if arrival_stop == CONTROLLED_STOPS[-1]:
                        self.bool_terminal_state = True
                        self.fixed_stop_unload()
                        self._add_observations()
                        self.fixed_stop_arrivals()
                        self.fixed_stop_load()
                        self.fixed_stop_depart()
                    else:
                        self.bool_terminal_state = False
                        self.fixed_stop_unload()
                        self.fixed_stop_arrivals()
                        self._add_observations()
                    return False
            self.fixed_stop_unload()
            self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return self.prep()

        if self.event_type == 0:
            self.terminal_departure()
            return self.prep()

