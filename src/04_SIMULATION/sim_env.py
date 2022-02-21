from input import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from agents_sim import Passenger, Stop, Trip, Bus, Log, TripLog
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
        self.trip_log = []

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
            link_time_params_light = (link_time_params[0]*0.6, link_time_params[1], link_time_params[2])
            runtime = lognorm.rvs(*link_time_params_light)
            # minim, maxim = link_time_extremes
            # if runtime > maxim:
            #     runtime = min(1.2*maxim, runtime)
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
            self.trip_log.append(TripLog(trip_id, STOPS))
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
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS[curr_stop_idx]] = bus.arr_t
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
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS[curr_stop_idx]] = bus.dep_t
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 2 if bus.next_stop_id == STOPS[-1] else 1
        if self.no_overtake_policy:
            bus.next_event_time = deepcopy(self.no_overtake())

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def inbound_ready_to_dispatch(self):
        bus = self.bus
        bus.last_stop_id = STOPS[0]
        bus.next_stop_id = STOPS[1]
        bus.arr_t = self.time
        self.stops[0].last_arr_times.append(self.time)
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS[0]] = bus.arr_t
        return

    def inbound_dispatch(self, hold=0):
        bus = self.bus
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

        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS[0]] = bus.dep_t

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
        self.log.recorded_arrivals[self.bus.active_trip[0].trip_id] = bus.arr_t
        trip_idx = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        self.trip_log[trip_idx].stop_arr_times[STOPS[-1]] = bus.arr_t
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
                # if trip.trip_id == 911879020:
                #     trip_idx = TRIP_IDS_IN.index(911878020)
                #     dep = self.trip_log[trip_idx].stop_dep_times[STOPS[0]]
                #     arr = self.trip_log[trip_idx].stop_arr_times[STOPS[-1]]
                    # print('-------')
                    # print(f'departure: {dep}')
                    # print(f'arrival: {arr}')
                    # print(f'run time: {arr-dep}')
                    # lst = []
                    # for k in self.trip_log[trip_idx].stop_dep_times:
                    #     lst.append((k, self.trip_log[trip_idx].stop_arr_times[k]))
                    #     print([k, self.trip_log[trip_idx].stop_arr_times[k], self.trip_log[trip_idx].stop_dep_times[k]])
                    # df = pd.DataFrame(lst,columns=['stop', 'time'])
                    # df.to_csv('in/trip_times' + str(random.uniform(0, 30)) + '.csv')
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
            trip_time_params = TRIP_TIMES1_PARAMS[interval]
            run_time = lognorm.rvs(*trip_time_params)
            arr_time = self.time + run_time
        else:
            assert route_type == 2
            interval = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            trip_time_params = TRIP_TIMES2_PARAMS[interval]
            run_time = lognorm.rvs(*trip_time_params)
            arr_time = self.time + run_time
        bus.dep_t = self.time
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        self.log.recorded_departures[trip.trip_id] = bus.dep_t
        return

    def no_overtake(self):
        next_instance_t = deepcopy(self.bus.next_event_time)
        trip_idx = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if trip_idx:
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
            active_buses = [bus for bus in self.buses if bus.active_trip]
            leading_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]
            if leading_bus:
                if self.bus.next_stop_id == leading_bus[0].next_stop_id:
                    next_instance_t = max(next_instance_t, leading_bus[0].next_event_time + NO_OVERTAKE_BUFFER)
        return next_instance_t

    def smart_dispatch(self):
        # check if previous trip has departed by the ready to dispatch time of the current trip else SWITCH
        bus = self.bus
        trip = bus.pending_trips[0]
        trip_idx = TRIP_IDS_IN.index(trip.trip_id)
        if trip_idx:
            dispatch_ready_time = max(self.time, trip.sched_time)
            # when only considering itself, dispatch time depends on the available time of the bus and scheduled time
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
            # in case the previous trip's bus is still making its way to the terminal this is how we will find it
            pending_buses = [bus for bus in self.buses if bus.pending_trips]
            pending_neighbor = [bus for bus in pending_buses if bus.pending_trips[0].trip_id == prev_trip_id]

            # in case the previous trip's bus is in the terminal waiting for its dispatch this is how we will find it
            # this can affect dispatch time because the twice previous trip bus could be delayed
            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]

            assert pending_neighbor or active_neighbor
            assert not (pending_neighbor and active_neighbor)

            if pending_neighbor:
                assert pending_neighbor[0].next_event_type == 4
                # CHECK CONDITION IF THE READY FOR DISPATCH TIME OF PREVIOUS TRIP'S BUS IS AFTER THE READY FOR DISPATCH
                # OF CURRENT TRIP
                if pending_neighbor[0].next_event_time > dispatch_ready_time:
                    # SWITCH PENDING TRIPS OF CURRENT BUS WITH THE PENDING TRIPS OF THE
                    # BUS (IR)RESPONSIBLE FOR THE PREVIOUS TRIP
                    trips_current_bus = deepcopy(self.bus.pending_trips)
                    id_current_bus = deepcopy(self.bus.bus_id)
                    trips_pending_neighbor = deepcopy(pending_neighbor[0].pending_trips)
                    id_pending_neighbor = deepcopy(pending_neighbor[0].bus_id)

                    self.bus.bus_id = id_pending_neighbor
                    self.bus.pending_trips = trips_pending_neighbor

                    pending_neighbor[0].bus_id = id_current_bus
                    pending_neighbor[0].pending_trips = trips_current_bus
                    return True
        return False

    def outbound_arrival(self):
        bus = self.bus
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            switched = self.smart_dispatch()
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            bus.last_stop_id = STOPS[0]
            bus.next_event_time = max(self.time, bus.active_trip[0].sched_time)
            bus.next_event_type = 0
        bus.arr_t = self.time
        self.log.recorded_arrivals[self.bus.finished_trips[-1].trip_id] = bus.arr_t
        return

    def smart_dispatch_initialized(self):
        trip = self.bus.active_trip[0]
        trip_idx = TRIP_IDS_IN.index(trip.trip_id)
        if trip_idx:
            # this is only for second trip onwards
            # we focus on switching the blocks if necessary
            # this function is for those trips that are initialized (at least we are at it's scheduled time)
            # so the only condition that matters is
            # has the previous trip departed? that's a no if it is still on the pending trip of some bus
            prev_trip_id = TRIP_IDS_IN[trip_idx - 1]

            pending_buses = [bus for bus in self.buses if bus.pending_trips]
            pending_neighbor = [bus for bus in pending_buses if bus.pending_trips[0].trip_id == prev_trip_id]

            active_buses = [bus for bus in self.buses if bus.active_trip]
            active_neighbor = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]

            assert pending_neighbor or active_neighbor
            assert not (pending_neighbor and active_neighbor)

            if pending_neighbor:
                # SWITCH ACTIVE TRIP + PENDING TRIPS OF CURRENT BUS WITH THE PENDING TRIPS OF THE
                # BUS (IR)RESPONSIBLE FOR THE PREVIOUS TRIP
                trips_current_bus = self.bus.active_trip[0] + self.bus.pending_trips
                id_current_bus = deepcopy(self.bus.bus_id)
                trips_pending_neighbor = deepcopy(pending_neighbor[0].pending_trips)
                id_pending_neighbor = deepcopy(pending_neighbor[0].bus_id)

                self.bus.bus_id = id_pending_neighbor
                self.bus.active_trip[0] = trips_pending_neighbor[0]
                if len(trips_pending_neighbor) > 1:
                    self.bus.pending_trips = trips_pending_neighbor[1:]
                else:
                    self.bus.pending_trips = []

                pending_neighbor[0].bus_id = id_current_bus
                pending_neighbor[0].pending_trips = trips_current_bus
        return

    def chop_pax(self):
        for p in self.completed_pax.copy():
            if p.trip_id not in FOCUS_TRIPS:
                self.completed_pax.remove(p)
        return

    def next_event(self):
        active_buses = [bus for bus in self.buses if bus.active_trip]
        if active_buses:
            next_event_times = [bus.next_event_time for bus in active_buses]
            min_event_time = min(next_event_times)
            min_event_time_idxs = [i for i, x in enumerate(next_event_times) if x == min_event_time]
            assert min_event_time_idxs
            if len(min_event_time_idxs) == 1:
                # self.bus = min(active_buses, key=lambda bus: bus.next_event_time)
                self.bus = active_buses[min_event_time_idxs[0]]
            else:
                coinciding_buses = [active_buses[i] for i in min_event_time_idxs]
                coinciding_dispatch_buses = [bus for bus in coinciding_buses if bus.next_event_type == 0]
                if len(coinciding_dispatch_buses) > 1:
                    # HANDLE THE EDGE CASE IN WHICH TWO (OR MORE)
                    # INBOUND DISPATCHES ARE SET AT THE SAME TIME
                    # THIS SCENARIO COULD SLIP THROUGH THE CRACKS IF
                    # BOTH TRIPS' OUTBOUND ARRIVAL HAPPENED AT THE SAME TIME
                    # AND BOTH WERE LATE: IN WHICH CASE
                    # READY FOR DISPATCH TIME = MAX(ARRIVAL TIME, SCHED TIME) = ARRIVAL TIME
                    # YOU SET AS A PRIORITY TO CALL INBOUND DISPATCH FOR THE FIRST SCHEDULED TRIP
                    trip_idxs = [TRIP_IDS_IN.index(cdb.active_trip[0].trip_id) for cdb in coinciding_dispatch_buses]
                    min_trip_idx = min(trip_idxs)
                    self.bus = coinciding_dispatch_buses[trip_idxs.index(min_trip_idx)]
                else:
                    # self.bus = min(active_buses, key=lambda bus: bus.next_event_time)
                    self.bus = active_buses[min_event_time_idxs[0]]

            self.time = float(self.bus.next_event_time)
            return False
        else:
            return True

    def prep(self):
        done = self.next_event()
        last_focus_bus = [bus for bus in self.buses if bus.bus_id == LAST_FOCUS_TRIP_BLOCK]
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in last_focus_bus[0].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                self.smart_dispatch_initialized()
            self.inbound_ready_to_dispatch()
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
        for trip_id in self.log.recorded_departures.copy():
            if trip_id not in (FOCUS_TRIP_IDS_OUT_SHORT + FOCUS_TRIP_IDS_OUT_LONG):
                self.log.recorded_departures.pop(trip_id)
                self.log.recorded_arrivals.pop(trip_id)
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
                if not next_dispatch_bus:
                    print(f'time is {str(timedelta(seconds=round(self.time)))}')
                    print(f'trip {TRIP_IDS_IN[idx_trip_id]} arriving at stop {stop.stop_id}')
                    print(f'looking for {next_trip_id} in {[bus.pending_trips[0].trip_id for bus in pending_bus]}')
                    print(f'previous departures {self.log.recorded_departures}')
                    print(f'previous arrivals {self.log.recorded_arrivals}')
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
            if backward_headway < 0.0:
                backward_headway = 0.0
        return backward_headway


class DetailedSimulationEnvWithControl(DetailedSimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnvWithControl, self).__init__(*args, **kwargs)

    def decide_bus_holding(self):
        # CHANGE------------
        # IT SHOULD BE HOLD TIME = MAX(0, MIN(PREV_ARR_T + LIMIT_HOLD - SELF.TIME, (BW_H-FW_H)/2 - SELF.TIME)
        # for previous trip
        stop = self.stops[STOPS.index(self.bus.last_stop_id)]
        backward_headway = self.get_backward_headway()
        forward_headway = self.time - stop.last_arr_times[-2]
        limit_holding = max(0, self.time - (stop.last_arr_times[-2] + MIN_ALLOWED_HW))
        if stop.stop_id == STOPS[0]:
            if backward_headway > forward_headway:
                holding_time = min(limit_holding, (backward_headway - forward_headway)/2)
                self.inbound_dispatch(hold=holding_time)
            else:
                self.inbound_dispatch()

        else:
            self.fixed_stop_load()
            if backward_headway > forward_headway:
                holding_time = min(limit_holding, (backward_headway - forward_headway)/2)
                self.fixed_stop_depart(hold=holding_time)
            else:
                self.fixed_stop_depart()
        return

    def prep(self):
        done = self.next_event()
        last_focus_bus = [bus for bus in self.buses if bus.bus_id == LAST_FOCUS_TRIP_BLOCK]
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in last_focus_bus[0].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                self.smart_dispatch_initialized()
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.inbound_ready_to_dispatch()
                    self.decide_bus_holding()
                    return False
            self.inbound_ready_to_dispatch()
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
            self.trip_log.append(TripLog(trip_id, STOPS))
        for trip_id in CONTROL_TRIP_IDS:
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
                        p.wait_time = 0.0
                        bus.pax.append(p)
                        self.stops[curr_stop_idx].pax.remove(p)
                        bus.ons += 1
                    else:
                        p.denied = 1
                        bus.denied += 1
                else:
                    break
        self.stops[curr_stop_idx].last_dep_times.append(deepcopy(bus.dep_t))
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS[curr_stop_idx]] = bus.dep_t
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

    def delayed_reward(self, s0, s1, neighbor_prev_hold, weight_ride=0):
        bus = self.bus
        s0_idx = STOPS.index(s0) # you check from those pax boarding next to the control stop (impacted)
        s1_idx = STOPS.index(s1) # next control point
        trip_id = bus.active_trip[0].trip_id
        trip_idx = TRIP_IDS_IN.index(trip_id)

        # FRONT NEIGHBOR'S REWARD
        agent_trip_idx = trip_idx - 1
        agent_trip_id = TRIP_IDS_IN[agent_trip_idx]
        # agent bus not considered responsible for the wait time impact in the control point upon arrival to it
        # only responsible for the impact on wait time at the control point for the following trip
        # and the riding time impact in case of holding which is accounted for later
        # for example if there was skipping at that stop previously the penalty would otherwise be tremendous
        # despite having nothing to do with that decision
        stop_idx_before_set = [s for s in range(0, s0_idx)]
        stop_idx_set = [s for s in range(s0_idx, s1_idx)]
        pax_count = 0
        # NOW TWO SCENARIOS: 1. THE FRONT BUS FINISHED ITS TRIP AND THEREFORE ALL PAX INFO SHOULD BE IN COMPLETED PAX
        # OR 2. IT IS STILL ACTIVE AND SOME PAX INFO MAY BE INCLUDED INSIDE BUS ATTRIBUTE PAX
        agent_board_count = 0
        sum_rew_agent_wait_time = 0
        sum_rew_agent_ride_time = 0

        for pax in self.completed_pax:
            if pax.trip_id == agent_trip_id and pax.orig_idx in stop_idx_set:
                pax_count += 1
                wait = pax.wait_time
                ride = pax.alight_time - pax.board_time
                assert ride > 0
                if pax.orig_idx == stop_idx_set[0] and pax.wait_time != 0.0:
                    wait += neighbor_prev_hold
                sum_rew_agent_wait_time += wait
                sum_rew_agent_ride_time += ride
                agent_board_count += 1
            elif pax.trip_id == agent_trip_id and pax.dest_idx in stop_idx_set[1:]:
                pax_count += 1
                ride = pax.alight_time - self.trip_log[agent_trip_idx].stop_arr_times[s0]
                assert ride > 0
                sum_rew_agent_ride_time += ride
        active_buses = [bus for bus in self.buses if bus.active_trip]
        front_active_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == agent_trip_id]
        if front_active_bus:
            for pax in front_active_bus[0].pax:
                if pax.orig_idx in stop_idx_set:
                    pax_count += 1
                    wait = pax.wait_time
                    ride = self.trip_log[agent_trip_idx].stop_arr_times[s1] - pax.board_time
                    assert ride > 0
                    if pax.orig_idx == stop_idx_set[0] and pax.wait_time != 0.0:
                        wait += neighbor_prev_hold
                    sum_rew_agent_wait_time += wait
                    sum_rew_agent_ride_time += ride
                    agent_board_count += 1
                elif pax.orig_idx in stop_idx_before_set:
                    pax_count += 1
                    ride = self.trip_log[agent_trip_idx].stop_arr_times[s1] - self.trip_log[agent_trip_idx].stop_arr_times[s0]
                    assert ride > 0
                    sum_rew_agent_ride_time += ride

        # CURRENT TRIP'S REWARD CONTRIBUTION
        stop_idx_set = [s for s in range(s0_idx, s1_idx)]
        # +1 is to catch the index of the control point only if it will be used to update
        sum_rew_behind_wait_time = 0
        sum_rew_behind_ride_time = 0
        for pax in bus.pax:
            if pax.orig_idx in stop_idx_set:
                pax_count += 1
                sum_rew_behind_wait_time += pax.wait_time
                ride = self.trip_log[trip_idx].stop_arr_times[s1] - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.orig_idx in stop_idx_before_set:
                pax_count += 1
                ride = self.trip_log[trip_idx].stop_arr_times[s1] - self.trip_log[trip_idx].stop_arr_times[s0]
                assert ride > 0
                sum_rew_behind_ride_time += ride
        for pax in self.completed_pax:
            if pax.trip_id == trip_id and pax.orig_idx in stop_idx_set:
                pax_count += 1
                sum_rew_behind_wait_time += pax.wait_time
                ride = pax.alight_time - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.trip_id == trip_id and pax.dest_idx in stop_idx_set[1:]:
                pax_count += 1
                ride = pax.alight_time - self.trip_log[trip_idx].stop_arr_times[s0]
                assert ride > 0
                sum_rew_behind_ride_time += ride
        reward_wait = (sum_rew_behind_wait_time + sum_rew_agent_wait_time)
        reward_ride = (sum_rew_behind_ride_time + sum_rew_agent_ride_time)
        reward = - (reward_wait + weight_ride * reward_ride) / 60 / 60
        # in hours
        return reward

    def update_rewards(self, simple_reward=False, weight_ride_time=0):
        bus = self.bus
        trip_id = bus.active_trip[0].trip_id

        if trip_id != CONTROL_TRIP_IDS[0] and bus.last_stop_id != CONTROLLED_STOPS[0]:
            # KEY STEP: WE NEED TO DO TWO THINGS:

            # ONE IS TO FILL IN THE AGENT'S REWARD FOR THE PREVIOUS STEP (IF THERE IS A PREVIOUS STEP)

            # TWO IS TO FILL THE FRONT NEIGHBOR'S REWARD FOR ITS PREVIOUS STEP (IF THERE IS AN ACTIVE FRONT NEIGHBOR)
            # TWO ALSO TRIGGERS SENDING THE SARS TUPLE INTO THE POOL OF COMPLETED EXPERIENCES
            k = CONTROLLED_STOPS.index(bus.last_stop_id)
            curr_control_stop = CONTROLLED_STOPS[k]
            prev_control_stop = CONTROLLED_STOPS[k - 1]
            trip_idx = TRIP_IDS_IN.index(trip_id)
            neighbor_trip_id = TRIP_IDS_IN[trip_idx - 1]
            sars_idx = k - 1
            assert sars_idx >= 0
            assert sars_idx < len(self.trips_sars[neighbor_trip_id])
            assert sars_idx < len(self.trips_sars[trip_id])
            if simple_reward:
                prev_sars = self.trips_sars[trip_id][sars_idx]
                # fw_h0 = prev_sars[0][IDX_FW_H]
                fw_h1 = prev_sars[3][IDX_FW_H]
                planned_fw_h = SCHED_DEP_IN[trip_idx] - SCHED_DEP_IN[trip_idx - 1]
                reward = abs(fw_h1-planned_fw_h)
                self.trips_sars[trip_id][sars_idx][2] = reward
                self.pool_sars.append(self.trips_sars[trip_id][sars_idx] + [self.bool_terminal_state])
            else:
                prev_sars = self.trips_sars[neighbor_trip_id][sars_idx]
                prev_action = prev_sars[1]
                prev_hold = (prev_action - 1) * BASE_HOLDING_TIME if prev_action > 1 else 0
                reward = self.delayed_reward(prev_control_stop, curr_control_stop, prev_hold, weight_ride=weight_ride_time)
                assert self.trips_sars[neighbor_trip_id][sars_idx][3]
                self.trips_sars[neighbor_trip_id][sars_idx][2] = reward
                self.pool_sars.append(self.trips_sars[neighbor_trip_id][sars_idx] + [self.bool_terminal_state])
        return

    def prep(self):
        done = self.next_event()
        last_focus_bus = [bus for bus in self.buses if bus.bus_id == LAST_FOCUS_TRIP_BLOCK]
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in last_focus_bus[0].finished_trips]:
            return True
        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            if not self.bus.finished_trips:
                # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
                self.smart_dispatch_initialized()
            arrival_stop = STOPS[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.bool_terminal_state = False
                    self.inbound_ready_to_dispatch()
                    self._add_observations()
                    return False
            self.inbound_ready_to_dispatch()
            self.inbound_dispatch()
            return self.prep()

        if self.bus.next_event_type == 1:
            arrival_stop = self.bus.next_stop_id
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS:
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
