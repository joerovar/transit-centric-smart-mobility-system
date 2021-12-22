from input import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from classes_simul import Passenger, Stop, Trip, Bus
from datetime import timedelta


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
            if interv_idx >= TIME_NR_INTERVALS:
                print(str(timedelta(seconds=round(start_time))))
                print(str(timedelta(seconds=round(time_control))))
            mean_runtime = LINK_TIMES_MEAN[stop0 + '-' + stop1][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[stop0 + '-' + stop1]
        time_control += mean_runtime
        assert mean_runtime
    arrival_time = time_control
    return arrival_time


def _compute_reward(action, fw_h, bw_h, trip_id, prev_bw_h, prev_fw_h, prev_pax_at_s):
    # trip_idx = TRIP_IDS_IN.index(trip_id)
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
        # self.next_actual_departures = []
        # self.next_trip_ids = []
        # self.active_trips = []
        # self.last_stop = []
        # self.next_stop = []
        # self.dep_t = []
        # self.arr_t = []
        # self.offs = []
        # self.ons = []
        # self.pax_at_stop = []
        # self.denied = []
        # self.load = []
        # self.bus_idx = 0
        # self.event_type = 0
        # self.next_instance_time = []
        self.time = 0.0
        # self.last_bus_time = {}
        # self.track_denied_boardings = {}
        # RECORDINGS
        self.trajectories = {}


    # def create_trip(self):
    #     trip_file = [
    #         self.next_instance_time, self.last_stop, self.next_stop, self.load,
    #         self.dep_t, self.arr_t, self.ons, self.offs, self.denied, self.pax_at_stop
    #     ]
    #     for tf in trip_file:
    #         tf.append(0)
    #     return
    #
    # def remove_trip(self):
    #     trip_file = [
    #         self.active_trips, self.next_instance_time, self.last_stop, self.next_stop, self.load,
    #         self.dep_t, self.arr_t, self.ons, self.offs, self.denied, self.pax_at_stop
    #     ]
    #     for tf in trip_file:
    #         tf.pop(self.bus_idx)
    #     return


class DetailedSimulationEnv(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnv, self).__init__(*args, **kwargs)
        self.stops = []
        # self.trips = []
        self.completed_pax = []
        self.buses = []
        self.bus = Bus(0)

    def record_trajectories(self, pickups=0, offs=0, denied_board=0, hold=0, skip=False):
        bus = self.bus
        trip_id = bus.active_trip[0].trip_id
        trajectory = [bus.last_stop_id, round(bus.arr_t, 1), round(bus.dep_t, 1),
                      len(bus.pax), pickups, offs, denied_board, hold, int(skip)]
        self.trajectories[trip_id].append(trajectory)
        return

    def get_travel_time(self):
        # i = self.bus_idx
        # prev_stop = self.last_stop[i]
        # next_stop = self.next_stop[i]
        bus = self.bus
        link = str(bus.last_stop_id) + '-' + str(bus.next_stop_id)
        if self.time_dependent_travel_time:
            interv = get_interval(self.time, TIME_INTERVAL_LENGTH_MINS)
            interv_idx = interv - TIME_START_INTERVAL
            mean_runtime = LINK_TIMES_MEAN[link][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[link]
        assert mean_runtime
        s = LOGN_S
        runtime = lognorm.rvs(s, scale=mean_runtime)
        return runtime

    def no_overtake(self):
        # if this is the first bus then ignore and return
        # i is subject bus index and j = i - 1 (leading bus)
        active_buses = [bus for bus in self.buses if bus.active_trip and bus != self.bus]
        active_neighbors = [bus for bus in active_buses if bus.active_trip[0].route_type == self.bus.active_trip[0].route_type]
        bus = self.bus
        next_instance_t = deepcopy(bus.next_event_time)
        for neighbor in active_neighbors:
            if bus.next_stop_id == neighbor.next_stop_id:
                next_instance_t = neighbor.next_event_time + NO_OVERTAKE_BUFFER
                break
        return next_instance_t

    def reset_simulation(self):
        self.time = START_TIME_SEC
        # self.bus_idx = 0
        self.bus = Bus(0)
        # for records
        self.trajectories = {}
        for trip_id in TRIP_IDS_IN:
            self.trajectories[trip_id] = []

        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            self.buses.append(Bus(block_id))
            trip_set = block_trip_set[1]
            for trip_info in trip_set:
                trip_id = trip_info[0]
                sched_time = trip_info[1]
                route_type = trip_info[2]
                self.buses[-1].pending_trips.append(Trip(trip_id, sched_time, route_type))
        # self.trips = []
        # for i in range(len(ORDERED_TRIPS)):
        #     self.trips.append(Trip(ORDERED_TRIPS[i]))
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            if trip.route_type == 0:
                bus.last_stop_id = STOPS[0]
            random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.sched_time + random_delay
            bus.next_event_type = 3 if trip.route_type else 0
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
                for k in range(POST_PROCESSED_DEM_START_INTERVAL, POST_PROCESSED_DEM_END_INTERVAL):
                    start_edge_interval = k * DEM_INTERVAL_LENGTH_MINS * 60
                    end_edge_interval = start_edge_interval + DEM_INTERVAL_LENGTH_MINS * 60
                    od_rate = ODT[k - POST_PROCESSED_DEM_START_INTERVAL, i, j]
                    if not np.isnan(od_rate):
                        max_size = int(od_rate * (DEM_INTERVAL_LENGTH_MINS / 60) * 3)
                        if od_rate > 0:
                            temp_pax_interarr_times = np.random.exponential(3600 / od_rate, size=max_size)
                            temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
                            if k == POST_PROCESSED_DEM_START_INTERVAL:
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
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.next_stop_id)
        bus.last_stop_id = STOPS[curr_stop_idx]
        bus.next_stop_id = STOPS[curr_stop_idx + 1]
        bus.arr_t = self.time
        bus.offs = 0
        for p in self.bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.bus.pax.remove(p)
                bus.offs += 1
        return

    # def fixed_stop_arrivals(self):
    #     i = self.bus_idx
    #     curr_stop_idx = STOPS.index(self.last_stop[i])
    #     self.pax_at_stop[i] = 0
    #     for p in self.stops[curr_stop_idx].pax.copy():
    #         if p.arr_time <= self.time:
    #             self.pax_at_stop[i] += 1
    #         else:
    #             break
    #     return

    def fixed_stop_load(self):
        bus = self.bus
        bus_load = len(bus.pax)
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        # curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        bus.denied = 0
        assert bus_load >= 0
        bus.ons = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + bus.ons + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break
        # self.load[i] += self.ons[i]
        return

    def fixed_stop_depart(self, hold=0):
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        stop = self.stops[curr_stop_idx]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME,
                             ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
        dwell_time = (bus.ons + bus.offs > 0) * dwell_time_pax
        if hold:
            dwell_time = max(hold, dwell_time_pax)
        bus.dep_t = self.time + dwell_time

        bus_load = len(bus.pax) - bus.ons
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= bus.dep_t:
                if bus_load + bus.ons + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break

        if self.no_overtake_policy and stop.last_bus_time:
            bus.dep_t = max(stop.last_bus_time + NO_OVERTAKE_BUFFER, bus.dep_t)
        stop.last_bus_time = deepcopy(bus.dep_t)

        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 2 if bus.next_stop_id == STOPS[-1] else 1
        if self.no_overtake_policy:
            bus.next_event_time = self.no_overtake()

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def inbound_dispatch(self, hold=0):
        # first create file for trip
        # i = self.bus_idx
        # self.create_trip()
        bus = self.bus
        bus.last_stop_id = STOPS[0]
        bus.next_stop_id = STOPS[1]
        # s = self.last_stop[i]
        bus.arr_t = self.time
        # curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        bus_load = len(bus.pax)
        bus.denied = 0
        stop = self.stops[0]
        for p in self.stops[0].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + bus.ons + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[0].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1

        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = ACC_DEC_TIME + bus.ons * BOARDING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.ons > 0) * dwell_time
        if hold:
            dwell_time = max(hold, dwell_time)
        bus.dep_t = self.time + dwell_time
        stop.last_bus_time = deepcopy(bus.dep_t)
        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        if self.no_overtake_policy:
            bus.next_event_time = self.no_overtake()
        bus.next_event_type = 1
        return

    def inbound_arrival(self):
        bus = self.bus
        # i = self.bus_idx
        bus.arr_t = self.time
        curr_stop_idx = STOPS.index(bus.next_stop_id)
        # curr_trip_idx = TRIP_IDS_IN.index(self.active_trips[i])
        for p in bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                bus.pax.remove(p)
                bus.offs += 1
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = ACC_DEC_TIME + bus.offs * ALIGHTING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.offs > 0) * dwell_time

        assert len(bus.pax) == 0
        bus.last_stop_id = STOPS[curr_stop_idx]
        bus.dep_t = float(bus.arr_t) + dwell_time
        self.record_trajectories(offs=bus.offs)
        # delete record of trip
        # self.remove_trip()
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            route_type = trip.route_type
            # types {1: long, 2: short}
            if route_type == 1:
                next_dep_time = max(bus.dep_t, trip.sched_time)
            else:
                assert route_type == 2
                interval = get_interval(bus.dep_t, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
                mean, std = DEADHEAD_TIME_PARAMS[interval]
                deadhead_time = norm.rvs(loc=mean, scale=std)
                next_dep_time = max(bus.dep_t + deadhead_time, trip.sched_time)
            bus.next_event_time = next_dep_time
            bus.next_event_type = 3
        return

    def outbound_dispatch(self):
        bus = self.bus
        trip = bus.active_trip[0]
        route_type = trip.route_type
        # types {1: long, 2: short}
        if route_type == 1:
            interval = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            mean, std = TRIP_TIMES1_PARAMS[interval]
            run_time = norm.rvs(loc=mean, scale=std)
            arr_time = self.time + run_time
        else:
            assert route_type == 2
            interval = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            mean, std = TRIP_TIMES2_PARAMS[interval]
            run_time = norm.rvs(loc=mean, scale=std)
            arr_time = self.time + run_time
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        return

    def outbound_arrival(self):
        bus = self.bus
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)

            trip = bus.active_trip[0]
            bus.last_stop_id = STOPS[0]
            bus.next_event_time = max(self.time, trip.sched_time)
            bus.next_event_type = 0
        return

    def chop_pax(self):
        for p in self.completed_pax.copy():
            if p.trip_id not in FOCUS_TRIPS:
                self.completed_pax.remove(p)
        return

    def next_event(self):
        active_buses = [bus for bus in self.buses if bus.active_trip]
        if active_buses:
            self.bus = min(active_buses, key=lambda bus: bus.next_event_time)
            self.time = float(self.bus.next_event_time)
            # print('-------')
            # print(self.bus.next_event_type)
            # print(self.bus.next_event_time)
            return False
        else:
            return True

        # if self.next_actual_departures:
        #     next_departure = self.next_actual_departures[0]
        #     if False not in [next_departure < t for t in self.next_instance_time]:
        #         self.time = self.next_actual_departures[0]
        #         self.active_trips.append(self.next_trip_ids[0])
        #         self.bus_idx = -1
        #         self.event_type = 0
        #         self.next_actual_departures.pop(0)
        #         self.next_trip_ids.pop(0)
        #         return
        # if self.next_instance_time:
        #     self.time = min(self.next_instance_time)
        #     self.bus_idx = self.next_instance_time.index(self.time)
        #     next_stop = self.next_stop[self.bus_idx]
        #     assert next_stop != STOPS[0]
        #     if next_stop == STOPS[-1]:
        #         self.event_type = 2
        #     else:
        #         self.event_type = 1
        #     return

    def prep(self):
        done = self.next_event()
        # if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
        #     return True
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            self.fixed_stop_unload()
            # self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.bus.next_event_type == 2:
            self.inbound_arrival()
            return False

        if self.bus.next_event_type == 3:
            self.outbound_dispatch()
            return False

        if self.bus.next_event_type == 4:
            self.outbound_arrival()
            return False

    def chop_trajectories(self):
        for trip in self.trajectories.copy():
            if trip not in FOCUS_TRIPS:
                self.trajectories.pop(trip)
        return

    def process_results(self):
        self.chop_trajectories()
        self.chop_pax()
        return


class DetailedSimulationEnvWithControl(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithControl, self).__init__(*args, **kwargs)

    def decide_bus_holding(self):
        # i = self.bus_idx
        # stop_id = self.last_stop[i]
        # trip_id = self.active_trips[i]
        # for previous trip
        stop = self.stops[STOPS.index(self.bus.last_stop_id)]
        forward_headway = self.time - stop.last_bus_time
        idx_trip_id = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if stop.stop_id == STOPS[0]:
            # terminal holding
            next_sched_dep_in = SCHED_DEP_IN[idx_trip_id + 1]
            backward_headway = next_sched_dep_in - self.time
            if backward_headway < 0:
                backward_headway = 0
            if backward_headway > forward_headway:
                holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
                self.inbound_dispatch(hold=holding_time)
            else:
                self.inbound_dispatch()
        else:
            next_trip_id = TRIP_IDS_IN[idx_trip_id + 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == next_trip_id]
            if active_neighbor:
                if active_neighbor[0].last_stop_id in STOPS:
                    dep_t = active_neighbor[0].dep_t
                    stop0 = active_neighbor[0].last_stop_id
                    stop1 = self.bus.last_stop_id
                else:
                    dep_t = SCHED_DEP_IN[idx_trip_id + 1]
                    stop0 = STOPS[0]
                    stop1 = self.bus.last_stop_id
            else:
                dep_t = SCHED_DEP_IN[idx_trip_id + 1]
                stop0 = STOPS[0]
                stop1 = self.bus.last_stop_id

            behind_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
            backward_headway = behind_trip_arrival_time - self.time
            if backward_headway < 0:
                backward_headway = 0
            # self.fixed_stop_arrivals()
            self.fixed_stop_load()
            if backward_headway > forward_headway:
                holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
                self.fixed_stop_depart(hold=holding_time)
            else:
                self.fixed_stop_depart()
        return

    def prep(self):
        done = self.next_event()
        # if (LAST_FOCUS_TRIP not in self.active_trips) & (LAST_FOCUS_TRIP not in self.next_trip_ids):
        #     return True
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if (trip_id != TRIP_IDS_IN[0]) & (trip_id != TRIP_IDS_IN[-1]):
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.decide_bus_holding()
                    return False
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            arrival_stop = self.bus.next_stop_id
            trip_id = self.bus.active_trip[0].trip_id
            if (trip_id != TRIP_IDS_IN[0]) & (trip_id != TRIP_IDS_IN[-1]):
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.fixed_stop_unload()
                    self.decide_bus_holding()
                    return False
            self.fixed_stop_unload()
            # self.fixed_stop_arrivals()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.bus.next_event_type == 2:
            self.inbound_arrival()
            return False

        if self.bus.next_event_type == 3:
            self.outbound_dispatch()
            return False

        if self.bus.next_event_type == 4:
            self.outbound_arrival()
            return False


class DetailedSimulationEnvWithDeepRL(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithDeepRL, self).__init__(*args, **kwargs)
        self.trips_sars = {}
        self.bool_terminal_state = False
        self.pool_sars = []

    def reset_simulation(self):
        self.time = START_TIME_SEC
        self.bool_terminal_state = False
        self.pool_sars = []
        self.bus = Bus(0)

        # for records
        self.trajectories = {}
        self.trips_sars = {}
        for trip_id in TRIP_IDS_IN:
            self.trajectories[trip_id] = []
            self.trips_sars[trip_id] = []

        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            self.buses.append(Bus(block_id))
            trip_set = block_trip_set[1]
            for trip_info in trip_set:
                trip_id = trip_info[0]
                sched_time = trip_info[1]
                route_type = trip_info[2]
                self.buses[-1].pending_trips.append(Trip(trip_id, sched_time, route_type))
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            if trip.route_type == 0:
                bus.last_stop_id = STOPS[0]
            random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.sched_time + random_delay
            bus.next_event_type = 3 if trip.route_type else 0
        # initialize passenger demand
        self.completed_pax = []
        self.initialize_pax_demand()
        return False

    def wait_time_reward(self, s0, s1, agent_bus=False, prev_hold=0, prev_load=0):
        bus = self.bus
        reward = 0
        trip_id = bus.active_trip[0].trip_id
        trip_idx = TRIP_IDS_IN.index(trip_id)
        scheduled_headway = SCHED_DEP_IN[trip_idx] - SCHED_DEP_IN[trip_idx-1]
        scheduled_wait = scheduled_headway / 2
        s0_idx = STOPS.index(s0) + 1 # you check from those pax boarding next to the control stop (impacted)
        s1_idx = STOPS.index(s1) # next control point
        stop_idx_set = [s for s in range(s0_idx, s1_idx+1)] # +1 is to catch the index of the control point
        for pax in bus.pax:
            if pax.orig_idx in stop_idx_set:
                excess_wait = max(0, pax.wait_time - scheduled_wait)
                reward += excess_wait
        for pax in self.completed_pax:
            if pax.trip_id == trip_id and pax.orig_idx in stop_idx_set:
                excess_wait = max(0, pax.wait_time - scheduled_wait)
                reward += excess_wait
        if agent_bus:
            # hold is from sars and prev load from prev state
            reward += prev_hold * prev_load
            # add impact on passengers on board
        # print('------')
        # print([s0, s1, stop_idx_set])
        # print([max(pax.wait_time-scheduled_wait,0) for pax in bus.pax if pax.orig_idx in stop_idx_set])
        # print([max(pax.wait_time,0) for pax in self.completed_pax if pax.orig_idx in stop_idx_set and pax.trip_id == trip_id])
        # print(scheduled_wait)
        # print(reward)
        reward = (-reward / 60)
        # in minutes
        return reward

    def fixed_stop_load(self, skip=False):
        # i = self.bus_idx
        # bus_load = self.load[i]
        # s = self.last_stop[i]
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        # curr_trip_idx = ORDERED_TRIPS.index(self.active_trips[i])
        bus_load = len(bus.pax)
        assert bus_load >= 0
        bus.denied = 0
        bus.ons = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if bus_load + bus.ons + 1 <= CAPACITY and not skip:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break
        return

    def fixed_stop_depart(self, hold=0, skip=False):
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        stop = self.stops[curr_stop_idx]
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if skip:
            dwell_time = ACC_DEC_TIME + ALIGHTING_TIME * bus.offs + dwell_time_error
            dwell_time = (bus.offs > 0) * dwell_time
        elif hold:
            dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME, ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (bus.ons + bus.offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(bus.ons * BOARDING_TIME, bus.offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (bus.ons + bus.offs > 0) * dwell_time

        bus.dep_t = self.time + dwell_time

        bus_load = len(bus.pax) - bus.ons
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= bus.dep_t:
                if bus_load + bus.ons + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break

        if self.no_overtake_policy and stop.last_bus_time:
            bus.dep_t = max(stop.last_bus_time + NO_OVERTAKE_BUFFER, bus.dep_t)
        stop.last_bus_time = deepcopy(bus.dep_t)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime

        if self.no_overtake_policy:
            bus.next_event_time = self.no_overtake()
        bus.next_event_type = 2 if bus.next_stop_id == STOPS[-1] else 1
        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        stop = self.stops[STOPS.index(self.bus.last_stop_id)]
        forward_headway = self.time - stop.last_bus_time
        idx_trip_id = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if stop.stop_id == STOPS[0]:
            # terminal holding
            next_sched_dep_in = SCHED_DEP_IN[idx_trip_id + 1]
            backward_headway = next_sched_dep_in - self.time
            if backward_headway < 0:
                backward_headway = 0
        else:
            next_trip_id = TRIP_IDS_IN[idx_trip_id + 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == next_trip_id]
            if active_neighbor:
                dep_t = active_neighbor[0].dep_t
                stop0 = active_neighbor[0].last_stop_id
                stop1 = self.bus.last_stop_id
            else:
                dep_t = SCHED_DEP_IN[idx_trip_id + 1]
                stop0 = STOPS[0]
                stop1 = self.bus.last_stop_id

            behind_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
            backward_headway = behind_trip_arrival_time - self.time
            if backward_headway < 0:
                backward_headway = 0

        stop_idx = STOPS.index(self.bus.last_stop_id)
        trip_id = self.bus.active_trip[0].trip_id
        trip_sars = self.trips_sars[trip_id]
        bus_load = len(self.bus.pax)

        route_progress = stop_idx/len(STOPS)
        pax_at_stop = 0
        for p in self.stops[stop_idx].pax.copy():
            if p.arr_time <= self.time:
                pax_at_stop += 1
            else:
                break
        new_state = [route_progress, bus_load, forward_headway, backward_headway, pax_at_stop]

        if trip_sars:
            self.trips_sars[trip_id][-1][3] = new_state
        if not self.bool_terminal_state:
            new_sars = [new_state, 0, 0, []]
            self.trips_sars[trip_id].append(new_sars)
        return

    def take_action(self, action):
        # i = self.bus_idx
        bus = self.bus
        # record action in sars
        trip_id = bus.active_trip[0].trip_id
        self.trips_sars[trip_id][-1][1] = action

        if action:
            if bus.last_stop_id == STOPS[0]:
                self.inbound_dispatch(hold=(action - 1) * BASE_HOLDING_TIME)
            else:
                self.fixed_stop_load()
                self.fixed_stop_depart(hold=(action - 1) * BASE_HOLDING_TIME)
        else:
            if bus.last_stop_id == STOPS[0]:
                self.inbound_dispatch()
            else:
                self.fixed_stop_load(skip=True)
                self.fixed_stop_depart(skip=True)
        return

    def update_rewards(self):
        bus = self.bus
        if bus.last_stop_id != CONTROLLED_STOPS[0]:
            trip_id = bus.active_trip[0].trip_id
            curr_control_stop_idx = CONTROLLED_STOPS.index(bus.last_stop_id)
            curr_control_stop = CONTROLLED_STOPS[curr_control_stop_idx]
            prev_control_stop = CONTROLLED_STOPS[curr_control_stop_idx - 1]
            # KEY STEP: WE NEED TO DO TWO THINGS:

            # ONE IS TO FILL IN THE AGENT'S REWARD FOR THE PREVIOUS STEP (IF THERE IS A PREVIOUS STEP)

            if len(self.trips_sars[trip_id]) > 1:
                prev_sars = self.trips_sars[trip_id][-2]
                prev_action = prev_sars[1]
                prev_hold = (prev_action - 1) * BASE_HOLDING_TIME if prev_action > 1 else 0
                prev_load = prev_sars[0][IDX_LOAD_RL]

                self.trips_sars[trip_id][-2][2] = self.wait_time_reward(prev_control_stop, curr_control_stop,
                                                                        agent_bus=True, prev_hold=prev_hold,
                                                                        prev_load=prev_load)
            # TWO IS TO FILL THE FRONT NEIGHBOR'S REWARD FOR ITS PREVIOUS STEP (IF THERE IS AN ACTIVE FRONT NEIGHBOR)
            # TWO ALSO TRIGGERS SENDING THE SARS TUPLE INTO THE POOL OF COMPLETED EXPERIENCES
            if trip_id != TRIP_IDS_IN[1]:
                stop_idx = STOPS.index(self.bus.last_stop_id)
                route_progress = stop_idx / len(STOPS)
                trip_idx = TRIP_IDS_IN.index(trip_id)
                neighbor_trip_id = TRIP_IDS_IN[trip_idx - 1]
                neighbor_sars_set = self.trips_sars[neighbor_trip_id]
                find_idx_sars = None
                for n in range(len(neighbor_sars_set)):
                    if neighbor_sars_set[n][3]:
                        route_progress_neighbor = neighbor_sars_set[n][3][IDX_RT_PROGRESS]
                        if round(route_progress_neighbor,1) == round(route_progress, 1):
                            find_idx_sars = n
                            break
                    # else:
                        # print(round(neighbor_sars_set[n][0][IDX_RT_PROGRESS],1))
                        # print(round(route_progress, 1))

                if find_idx_sars:
                    # print(find_idx_sars)
                    reward = self.wait_time_reward(prev_control_stop, curr_control_stop)
                    self.trips_sars[neighbor_trip_id][find_idx_sars][2] += reward
                    self.pool_sars.append(self.trips_sars[neighbor_trip_id][find_idx_sars] + [self.bool_terminal_state])
        return

    def prep(self):
        done = self.next_event()
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if (trip_id != TRIP_IDS_IN[0]) & (trip_id != TRIP_IDS_IN[-1]):
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.bool_terminal_state = False
                    # print([trip_id, self.trips_sars[trip_id]])
                    self._add_observations()
                    # print([trip_id, self.trips_sars[trip_id]])
                    return False
            self.inbound_dispatch()

            return self.prep()

        if self.bus.next_event_type == 1:
            arrival_stop = self.bus.next_stop_id
            trip_id = self.bus.active_trip[0].trip_id
            if (trip_id != TRIP_IDS_IN[0]) & (trip_id != TRIP_IDS_IN[-1]):
                if arrival_stop in CONTROLLED_STOPS:
                    # print([trip_id, self.trips_sars[trip_id]])
                    if arrival_stop == CONTROLLED_STOPS[-1]:
                        self.bool_terminal_state = True
                        self.fixed_stop_unload()
                        self._add_observations()
                        self.fixed_stop_load()
                        self.fixed_stop_depart()
                    else:
                        self.bool_terminal_state = False
                        self.fixed_stop_unload()
                        self._add_observations()
                    # print([trip_id, self.trips_sars[trip_id]])
                    return False
            self.fixed_stop_unload()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return self.prep()

        if self.bus.next_event_type == 2:
            self.inbound_arrival()
            return self.prep()

        if self.bus.next_event_type == 3:
            self.outbound_dispatch()
            return self.prep()

        if self.bus.next_event_type == 4:
            self.outbound_arrival()
            return self.prep()
