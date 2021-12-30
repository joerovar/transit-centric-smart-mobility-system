from input import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from classes_simul import Passenger, Stop, Trip, Bus, Log
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
            mean_runtime = LINK_TIMES_MEAN[stop0 + '-' + stop1][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[stop0 + '-' + stop1]
        time_control += mean_runtime
        assert mean_runtime
    arrival_time = time_control
    return arrival_time


def _compute_reward(action, fw_h, bw_h, prev_bw_h, prev_fw_h, prev_pax_at_s):

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
        self.time = 0.0
        # RECORDINGS
        self.trajectories = {}


class DetailedSimulationEnv(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnv, self).__init__(*args, **kwargs)
        self.stops = []
        self.completed_pax = []
        self.buses = []
        self.bus = Bus(0)
        self.log = Log(TRIP_IDS_IN + TRIP_IDS_OUT)

    def record_trajectories(self, pickups=0, offs=0, denied_board=0, hold=0, skip=False):
        bus = self.bus
        trip_id = bus.active_trip[0].trip_id
        trajectory = [bus.last_stop_id, round(bus.arr_t, 1), round(bus.dep_t, 1),
                      len(bus.pax), pickups, offs, denied_board, hold, int(skip)]
        self.trajectories[trip_id].append(trajectory)
        return

    def get_travel_time(self):
        # i = self.bus_idx
        bus = self.bus
        link = str(bus.last_stop_id) + '-' + str(bus.next_stop_id)
        interv = get_interval(self.time, TIME_INTERVAL_LENGTH_MINS)
        interv_idx = interv - TIME_START_INTERVAL
        if self.time_dependent_travel_time:
            link_time_params = LINK_TIMES_PARAMS[link][interv_idx]
            link_time_extremes = LINK_TIMES_EXTREMES[link][interv_idx]
            if type(link_time_params) is not tuple:
                if interv_idx == 0:
                    link_time_params = LINK_TIMES_PARAMS[link][interv_idx + 1]
                    link_time_extremes = LINK_TIMES_EXTREMES[link][interv_idx + 1]
        else:
            link_time_params = SINGLE_LINK_TIMES_PARAMS[link]
            link_time_extremes = SINGLE_LINK_TIMES_EXTREMES
        try:
            runtime = lognorm.rvs(*link_time_params)
            minim, maxim = link_time_extremes
            if runtime > maxim:
                runtime = min(1.2*maxim, runtime)
        except ValueError:
            print(f'trip id {bus.active_trip[0].trip_id}')
            print(f'link {link}')
            print(f'time {self.time}')
            raise
        except TypeError:
            print(f'trip id {bus.active_trip[0].trip_id}')
            print(f'link {link}')
            print(f'time {self.time}')
            print(link_time_params)
            print(f'interval {interv_idx}')
            raise
        return runtime

    def reset_simulation(self):
        self.time = START_TIME_SEC
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
        # initialize log of arrival/departures for delay analysis
        self.log = Log(TRIP_IDS_IN + TRIP_IDS_OUT)
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
        self.stops[curr_stop_idx].last_arr_times.append(bus.arr_t)
        bus.offs = 0
        for p in self.bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.bus.active_trip[0].completed_pax.append(p)
                self.bus.pax.remove(p)
                bus.offs += 1
        return

    def fixed_stop_load(self):
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if len(bus.pax) + 1 <= CAPACITY:
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

        if self.no_overtake_policy and stop.last_dep_times:
            bus.dep_t = max(stop.last_dep_times[-1] + NO_OVERTAKE_BUFFER, bus.dep_t)

        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= bus.dep_t:
                if len(bus.pax) + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(p.arr_time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[curr_stop_idx].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break
        self.stops[curr_stop_idx].last_dep_times.append(deepcopy(bus.dep_t))

        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 2 if bus.next_stop_id == STOPS[-1] else 1
        if self.no_overtake_policy:
            bus.next_event_time = deepcopy(self.no_overtake())

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def inbound_dispatch(self, hold=0):
        bus = self.bus
        bus.last_stop_id = STOPS[0]
        bus.next_stop_id = STOPS[1]
        bus.arr_t = self.time
        self.stops[0].last_arr_times.append(self.time)
        bus.denied = 0
        stop = self.stops[0]
        for p in self.stops[0].pax.copy():
            if p.arr_time <= self.time:
                if len(bus.pax) + 1 <= CAPACITY:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    p.wait_time = float(p.board_time - p.arr_time)
                    bus.pax.append(p)
                    self.stops[0].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break

        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = bus.ons * BOARDING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.ons > 0) * dwell_time
        if hold:
            dwell_time = max(hold, dwell_time)

        bus.dep_t = self.time + dwell_time

        if stop.last_dep_times and self.no_overtake_policy:
            bus.dep_t = max(bus.dep_t, stop.last_dep_times[-1] + NO_OVERTAKE_BUFFER)

        # FOR THOSE PAX WHO ARRIVE DURING THE DWELL TIME
        if bus.dep_t > self.time:
            for p in self.stops[0].pax.copy():
                if p.arr_time <= bus.dep_t:
                    if len(bus.pax) + 1 <= CAPACITY:
                        p.trip_id = bus.active_trip[0].trip_id
                        p.board_time = float(p.arr_time)
                        p.wait_time = float(p.board_time - p.arr_time)
                        bus.pax.append(p)
                        self.stops[0].pax.remove(p)
                        bus.ons += 1
                    else:
                        p.denied = 1
                        bus.denied += 1
                else:
                    break

        self.stops[0].last_dep_times.append(deepcopy(bus.dep_t))
        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=hold)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        if self.no_overtake_policy:
            bus.next_event_time = deepcopy(self.no_overtake())
        bus.next_event_type = 1

        self.log.recorded_departures[self.bus.active_trip[0].trip_id] = bus.dep_t
        return

    def inbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        curr_stop_idx = STOPS.index(bus.next_stop_id)
        for p in bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.bus.active_trip[0].completed_pax.append(p)
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
        self.log.recorded_arrivals[self.bus.finished_trips[-1].trip_id] = bus.arr_t
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
        bus.dep_t = self.time
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        self.log.recorded_departures[trip.trip_id] = bus.dep_t
        return

    def no_overtake_dispatch(self):
        # check if previous trip has departed (NO OVERTAKE AT DISPATCH)
        bus = self.bus
        trip = bus.active_trip[0]
        trip_idx = TRIP_IDS_IN.index(trip.trip_id)
        # when only considering itself, dispatch time depends on the available time of the bus and the scheduled time
        dispatch_time = max(self.time, trip.sched_time)
        if trip_idx:
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
            # in case the previous trip's bus is still making its way to the terminal this is how we will find it
            pending_buses = [bus for bus in self.buses if bus.pending_trips]
            pending_neighbor = [bus for bus in pending_buses if bus.pending_trips[0].trip_id == prev_trip_id]

            # in case the previous trip's bus is in the terminal waiting for its dispatch this is how we will find it
            # this can affect dispatch time because the twice previous trip bus could be delayed
            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]

            if pending_neighbor:
                prev_trip_expected_dispatch_time = pending_neighbor[0].next_event_time
                if prev_trip_expected_dispatch_time > dispatch_time:
                    dispatch_time = prev_trip_expected_dispatch_time

                # in the extreme case that the previous trip's previous trip is delayed, we can find it in pending buses
                if trip_idx > 1:
                    twice_prev_trip_id = TRIP_IDS_IN[trip_idx - 2]
                    pending_buses = [bus for bus in self.buses if bus.pending_trips]
                    pending_next_neighbor = [bus for bus in pending_buses if
                                             bus.pending_trips[0].trip_id == twice_prev_trip_id]
                    if pending_next_neighbor:
                        twice_prev_trip_expect_dispatch_time = pending_next_neighbor[0].next_event_time
                        if twice_prev_trip_expect_dispatch_time > dispatch_time:
                            dispatch_time = pending_next_neighbor[0].next_event_time + 2 * NO_OVERTAKE_BUFFER

            elif active_neighbor:
                if active_neighbor[0].next_event_type == 0:
                    prev_trip_expected_dispatch_time = active_neighbor[0].next_event_time
                    if prev_trip_expected_dispatch_time > dispatch_time:
                        dispatch_time = prev_trip_expected_dispatch_time + NO_OVERTAKE_BUFFER
        return dispatch_time

    def outbound_arrival(self):
        bus = self.bus
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            bus.last_stop_id = STOPS[0]
            bus.next_event_time = self.no_overtake_dispatch()
            bus.next_event_type = 0
        bus.arr_t = self.time
        self.log.recorded_arrivals[self.bus.finished_trips[-1].trip_id] = bus.arr_t
        return

    def no_overtake(self):
        next_instance_t = deepcopy(self.bus.next_event_time)
        trip_idx = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if trip_idx:
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            leading_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]
            # active_neighbors = [bus for bus in active_buses if bus.active_trip[0].route_type == self.bus.active_trip[0].route_type]
            if leading_bus:
                if self.bus.next_stop_id == leading_bus[0].next_stop_id:
                    next_instance_t = max(next_instance_t, leading_bus[0].next_event_time + NO_OVERTAKE_BUFFER)
        return next_instance_t

    def allow_initialized_dispatch(self):
        bus = self.bus
        trip = self.bus.active_trip[0]
        trip_idx = TRIP_IDS_IN.index(trip.trip_id)
        if trip_idx:
            # this is only for second trip onwards
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
            pending_buses = [bus for bus in self.buses if bus.pending_trips]
            pending_bus = [bus for bus in pending_buses if bus.pending_trips[0].trip_id == prev_trip_id]

            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]

            if pending_bus:
                prev_trip_expected_dispatch_time = pending_bus[0].next_event_time
                if prev_trip_expected_dispatch_time > self.time:
                    bus.next_event_time = prev_trip_expected_dispatch_time + NO_OVERTAKE_BUFFER
                    return False
            elif active_neighbor:
                if active_neighbor[0].next_event_type == 0:
                    prev_trip_expected_dispatch_time = active_neighbor[0].next_event_time
                    if prev_trip_expected_dispatch_time > self.time:
                        bus.next_event_time = prev_trip_expected_dispatch_time + NO_OVERTAKE_BUFFER
                        return False
        return True

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
            return False
        else:
            return True

    def prep(self):
        done = self.next_event()
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                allow_initialized_dispatch = self.allow_initialized_dispatch()
                if not allow_initialized_dispatch:
                    return False
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            self.fixed_stop_unload()
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

    def get_backward_headway(self):
        stop = self.stops[STOPS.index(self.bus.last_stop_id)]
        idx_trip_id = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if stop.stop_id == STOPS[0]:
            # terminal holding
            next_trip_id = TRIP_IDS_IN[idx_trip_id + 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            next_dispatch_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == next_trip_id]
            if not next_dispatch_bus:
                pending_bus = [bus for bus in active_buses if bus.pending_trips]
                if pending_bus:
                    next_dispatch_bus = [bus for bus in pending_bus if bus.pending_trips[0].trip_id == next_trip_id]
            next_sched_dep_t = SCHED_DEP_IN[idx_trip_id + 1]
            next_dep_t = max(next_dispatch_bus[0].next_event_time, next_sched_dep_t)
            backward_headway = next_dep_t - self.time
            assert backward_headway >= 0.0

        else:
            next_trip_id = TRIP_IDS_IN[idx_trip_id + 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == next_trip_id]
            if active_neighbor:
                dep_t = active_neighbor[0].dep_t
                stop0 = active_neighbor[0].last_stop_id
                if dep_t == 0:
                    assert stop0 == STOPS[0]
                    dep_t = SCHED_DEP_IN[idx_trip_id + 1]
            else:
                pending_bus = [bus for bus in active_buses if bus.pending_trips]
                next_dispatch_bus = [bus for bus in pending_bus if bus.pending_trips[0].trip_id == next_trip_id]
                if not next_dispatch_bus:
                    print(f'time is {str(timedelta(seconds=round(self.time)))}')
                    print(f'trip {TRIP_IDS_IN[idx_trip_id]} arriving at stop {stop.stop_id}')
                    print(f'looking for {next_trip_id} in {[bus.pending_trips[0].trip_id for bus in pending_bus]}')
                    print(f'previous departures {self.log.recorded_departures}')
                    print(f'previous arrivals {self.log.recorded_arrivals}')
                next_sched_dep_t = SCHED_DEP_IN[idx_trip_id + 1]
                dep_t = max(next_dispatch_bus[0].next_event_time, next_sched_dep_t)
                stop0 = STOPS[0]
            stop1 = self.bus.last_stop_id
            behind_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time)
            backward_headway = behind_trip_arrival_time - self.time
            if backward_headway < 0:
                backward_headway = 0
        return backward_headway


class DetailedSimulationEnvWithControl(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithControl, self).__init__(*args, **kwargs)

    def decide_bus_holding(self):
        # for previous trip
        stop = self.stops[STOPS.index(self.bus.last_stop_id)]
        forward_headway = self.time - stop.last_dep_times[-1]
        backward_headway = self.get_backward_headway()

        if stop.stop_id == STOPS[0]:
            if backward_headway > forward_headway:
                holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
                self.inbound_dispatch(hold=holding_time)
            else:
                self.inbound_dispatch()
        else:
            self.fixed_stop_load()
            if backward_headway > forward_headway:
                holding_time = min(LIMIT_HOLDING, backward_headway - forward_headway)
                self.fixed_stop_depart(hold=holding_time)
            else:
                self.fixed_stop_depart()
        return

    def prep(self):
        done = self.next_event()
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                allow_initialized_dispatch = self.allow_initialized_dispatch()
                if not allow_initialized_dispatch:
                    return False
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.decide_bus_holding()
                    return False
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            arrival_stop = self.bus.next_stop_id
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
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

    def wait_time_reward(self, s0, s1, agent_bus=False, prev_hold=0, prev_load=0, prev_pax_at_stop=0):
        bus = self.bus
        reward_wait_time = 0
        reward_ride_time = 0
        trip_id = bus.active_trip[0].trip_id
        trip_idx = TRIP_IDS_IN.index(trip_id)
        scheduled_headway = SCHED_DEP_IN[trip_idx] - SCHED_DEP_IN[trip_idx-1]
        scheduled_wait = scheduled_headway / 2
        s0_idx = STOPS.index(s0) # you check from those pax boarding next to the control stop (impacted)
        s1_idx = STOPS.index(s1) # next control point
        if agent_bus:
            # agent bus not considered responsible for the wait time impact in the control point upon arrival to it
            # only responsible for the impact on wait time at the control point for the following trip
            # and the riding time impact in case of holding which is accounted for later
            # for example if there was skipping at that stop previously the penalty would otherwise be tremendous
            # despite having nothing to do with that decision
            stop_idx_set = [s for s in range(s0_idx + 1, s1_idx)]
        else:
            stop_idx_set = [s for s in range(s0_idx + 1, s1_idx + 1)]
            # +1 is to catch the index of the control point only if it will be used to update
            # the reward of the front neighbor
        # print(f'looking for pax in the stop set {stop_idx_set}')
        # print(f'scheduled wait is {scheduled_wait}')
        for pax in bus.pax:
            if pax.orig_idx in stop_idx_set:
                excess_wait = max(0, pax.wait_time - scheduled_wait)
                reward_wait_time += excess_wait
        for pax in self.completed_pax:
            if pax.trip_id == trip_id and pax.orig_idx in stop_idx_set:
                excess_wait = max(0, pax.wait_time - scheduled_wait)
                reward_wait_time += excess_wait
        if agent_bus:
            # hold is from sars and prev load from prev state
            reward_ride_time += prev_hold * (prev_load + prev_pax_at_stop)
            # add impact on passengers on board
        reward = reward_ride_time + reward_wait_time
        reward = (-reward / 60)
        # in minutes
        return reward

    def fixed_stop_load(self, skip=False):
        bus = self.bus
        curr_stop_idx = STOPS.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time:
                if len(bus.pax) + 1 <= CAPACITY and not skip:
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

        if self.no_overtake_policy and stop.last_dep_times:
            bus.dep_t = max(stop.last_dep_times[-1] + NO_OVERTAKE_BUFFER, bus.dep_t)

        if not skip:
            for p in self.stops[curr_stop_idx].pax.copy():
                if p.arr_time <= bus.dep_t:
                    if len(bus.pax) + 1 <= CAPACITY:
                        p.trip_id = bus.active_trip[0].trip_id
                        p.board_time = float(p.arr_time)
                        p.wait_time = float(p.board_time - p.arr_time)
                        bus.pax.append(p)
                        self.stops[curr_stop_idx].pax.remove(p)
                        bus.ons += 1
                    else:
                        p.denied = 1
                        bus.denied += 1
                else:
                    break
        self.stops[curr_stop_idx].last_dep_times.append(deepcopy(bus.dep_t))
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime

        if self.no_overtake_policy:
            bus.next_event_time = deepcopy(self.no_overtake())
        bus.next_event_type = 2 if bus.next_stop_id == STOPS[-1] else 1
        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        curr_stop_idx = STOPS.index(self.bus.last_stop_id)
        stop = self.stops[curr_stop_idx]
        forward_headway = self.time - stop.last_dep_times[-1]
        backward_headway = self.get_backward_headway()

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
        if stop.stop_id == STOPS[0]:
            # then only inbound dispatch updates arr_times and dep_times so indexes are coordinated
            prev_fw_h = stop.last_arr_times[-1] - stop.last_dep_times[-2]
        else:
            # if it is an intermediate stop then the arrival time was already updated in fixed_stop_unload
            # so there is one extra arr_time than dep_time
            prev_fw_h = stop.last_arr_times[-2] - stop.last_dep_times[-2]
        new_state = [route_progress, bus_load, forward_headway, backward_headway, pax_at_stop, prev_fw_h]

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
            # print(prev_control_stop)
            # KEY STEP: WE NEED TO DO TWO THINGS:

            # ONE IS TO FILL IN THE AGENT'S REWARD FOR THE PREVIOUS STEP (IF THERE IS A PREVIOUS STEP)
            prev_sars = self.trips_sars[trip_id][-2]
            prev_action = prev_sars[1]
            prev_hold = (prev_action - 1) * BASE_HOLDING_TIME if prev_action > 1 else 0
            prev_load = prev_sars[0][IDX_LOAD_RL]
            prev_pax_at_stop = prev_sars[0][IDX_PAX_AT_STOP]
            # print('-----------')
            # print(f'time {str(timedelta(seconds=round(self.time)))}')
            # print(f'trip id {trip_id}')
            # print(f'about to handle immediate reward on action {prev_action} taken at {prev_control_stop} '
            #       f'yielding hold time {prev_hold}'
            #       f' on {prev_load} pax on bus and {prev_pax_at_stop} pax to board')
            self.trips_sars[trip_id][-2][2] = self.wait_time_reward(prev_control_stop, curr_control_stop,
                                                                    agent_bus=True, prev_hold=prev_hold,
                                                                    prev_load=prev_load,
                                                                    prev_pax_at_stop=prev_pax_at_stop)
            # TWO IS TO FILL THE FRONT NEIGHBOR'S REWARD FOR ITS PREVIOUS STEP (IF THERE IS AN ACTIVE FRONT NEIGHBOR)
            # TWO ALSO TRIGGERS SENDING THE SARS TUPLE INTO THE POOL OF COMPLETED EXPERIENCES
            if trip_id != TRIP_IDS_IN[2]:
                # print('accessed update of front neighbors reward!')
                stop_idx = STOPS.index(self.bus.last_stop_id)
                route_progress = stop_idx / len(STOPS)
                trip_idx = TRIP_IDS_IN.index(trip_id)
                neighbor_trip_id = TRIP_IDS_IN[trip_idx - 1]
                neighbor_sars_set = self.trips_sars[neighbor_trip_id]
                # print(f'this is front neighbors sars {neighbor_sars_set}')
                find_idx_sars = None
                # print(f'we will look for route progress {round(route_progress, 2)}')
                for n in range(len(neighbor_sars_set)):
                    if neighbor_sars_set[n][3]:
                        route_progress_neighbor = neighbor_sars_set[n][3][IDX_RT_PROGRESS]
                        # print(f'neighbor route progress to check is {round(route_progress_neighbor, 2)}')
                        if round(route_progress_neighbor, 2) == round(route_progress, 2):
                            # print(f'the same!')
                            find_idx_sars = n
                            break

                if find_idx_sars is not None:
                    # print(find_idx_sars)
                    # print('-----------')
                    # print(f'time {str(timedelta(seconds=round(self.time)))}')
                    # print(f'trip id {trip_id}')
                    # print(f'about to handle delayed reward on action {prev_action} taken at {prev_control_stop} '
                    #       f'yielding hold time {prev_hold}'
                    #       f' on {prev_load} pax on bus and {prev_pax_at_stop} pax to board')
                    reward = self.wait_time_reward(prev_control_stop, curr_control_stop)
                    self.trips_sars[neighbor_trip_id][find_idx_sars][2] += reward
                    # print(route_progress)
                    # print(self.trips_sars[neighbor_trip_id][find_idx_sars])
                    self.pool_sars.append(self.trips_sars[neighbor_trip_id][find_idx_sars] + [self.bool_terminal_state])
        return

    def prep(self):
        done = self.next_event()
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in self.buses[LAST_FOCUS_TRIP_BLOCK_IDX].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                allow_initialized_dispatch = self.allow_initialized_dispatch()
                if not allow_initialized_dispatch:
                    return self.prep()
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
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
            if trip_id not in NO_CONTROL_TRIP_IDS:
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
