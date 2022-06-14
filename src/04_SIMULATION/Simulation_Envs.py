from Inputs import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from Simulation_Classes import Passenger, Stop, Bus, TripLog


def get_interval(t, interval_length):
    # time in seconds, interval length in minutes
    interval = int(t / (interval_length * 60))
    return interval


def estimate_arrival_time(start_time, start_stop, end_stop, time_dependent_tt, current_time):
    temp_arr_t = start_time
    start_stop_idx = STOPS_OUTBOUND.index(start_stop)
    end_stop_idx = STOPS_OUTBOUND.index(end_stop)
    for i in range(start_stop_idx, end_stop_idx):
        stop0 = STOPS_OUTBOUND[i]
        stop1 = STOPS_OUTBOUND[i + 1]
        if time_dependent_tt:
            interv = get_interval(temp_arr_t, TIME_INTERVAL_LENGTH_MINS)
            if temp_arr_t > END_TIME_SEC:
                return np.nan
            interv_idx = interv - TIME_START_INTERVAL
            mean_runtime = LINK_TIMES_MEAN[stop0 + '-' + stop1][interv_idx]
        else:
            mean_runtime = SINGLE_LINK_TIMES_MEAN[stop0 + '-' + stop1]
        temp_arr_t = max(temp_arr_t + mean_runtime, current_time)
        assert mean_runtime
    arrival_time = temp_arr_t
    return arrival_time


def _compute_reward(action, fw_h, bw_h, prev_bw_h, prev_fw_h, prev_pax_at_s):
    hw_diff0 = abs(prev_fw_h - prev_bw_h)
    hw_diff1 = abs(fw_h - bw_h)
    reward = hw_diff0 - hw_diff1
    if prev_pax_at_s and action == SKIP_ACTION:
        reward -= 0.5*prev_bw_h
    return reward


class SimulationEnv:
    def __init__(self, time_dependent_travel_time=True, time_dependent_demand=True,
                 tt_factor=1.0, hold_adj_factor=0.0):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        # self.no_overtake_policy = no_overtake_policy
        self.time_dependent_travel_time = time_dependent_travel_time
        self.time_dependent_demand = time_dependent_demand
        self.time = 0.0
        # RECORDINGS
        self.trajectories_out = {}
        self.tt_factor = tt_factor
        self.hold_adj_factor = hold_adj_factor
        self.trajectories_in = {}


class DetailedSimulationEnv(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnv, self).__init__(*args, **kwargs)
        self.stops = []
        self.completed_pax = []
        self.buses = []
        self.bus = Bus(0, [])
        self.trip_log = []
        self.focus_trips_finished = []
        self.completed_pax_record = []
        self.out_trip_record = []
        self.in_trip_record = []

    def backward_headway(self):
        # terminal
        # search for active buses inbound and compute the time they will pick up pax at terminal (arrival time)
        expected_arr_t = []
        # this will collect the expected arrival time of all trips departing
        for bus in self.buses:
            # HERE WE TAKE TRIPS THAT ARE MID-ROUTE IN THE INBOUND DIRECTION
            if bus.active_trip and bus.active_trip[0].route_type in [1, 2] and bus.pending_trips:
                assert bus.next_event_type == 4 or bus.next_event_type == 3
                prev_dep_t = max(bus.dep_t, bus.active_trip[0].schedule[0])
                interv_idx = get_interval(prev_dep_t, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
                next_sched_t = bus.pending_trips[0].schedule[0]
                if bus.active_trip[0].route_type == 1:
                    estimated_in_run_t = np.mean(TRIP_T1_DIST_IN[interv_idx])
                else:
                    estimated_in_run_t = np.mean(TRIP_T2_DIST_IN[interv_idx])
                ready_time = prev_dep_t + estimated_in_run_t + MIN_LAYOVER_T
                if self.bus.last_stop_id == STOPS_OUTBOUND[0]:
                    expected_arr_t.append(max(ready_time, next_sched_t))
                else:
                    terminal_dep_t = max(ready_time, next_sched_t)
                    arr_t = estimate_arrival_time(terminal_dep_t, STOPS_OUTBOUND[0], self.bus.last_stop_id,
                                                  self.time_dependent_travel_time, self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? on buses on inbound')
                        expected_arr_t.append(arr_t)

            # HERE WE TAKE TRIPS THAT ARE IN THE TERMINAL WAITING TO BE DISPATCHED
            if bus.active_trip and bus.active_trip[0].route_type == 0 and bus.next_event_type == 0 and bus.bus_id != self.bus.bus_id:
                if self.bus.last_stop_id == STOPS_OUTBOUND[0]:
                    expected_arr_t.append(bus.next_event_time)
                else:
                    terminal_dep_t = bus.next_event_time
                    arr_t = estimate_arrival_time(terminal_dep_t, STOPS_OUTBOUND[0], self.bus.last_stop_id,
                                                  self.time_dependent_travel_time, self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? on garaged buses')
                        expected_arr_t.append(arr_t)

            # only for trips that are at a midpoint control route should check buses behind in the outbound direction

            if self.bus.last_stop_id != STOPS_OUTBOUND[0] and bus.active_trip and bus.active_trip[0].route_type == 0 and bus.next_event_type == 1:
                behind_last_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
                curr_stop_idx = STOPS_OUTBOUND.index(self.bus.last_stop_id)
                if behind_last_stop_idx < curr_stop_idx:
                    stop0 = bus.last_stop_id
                    stop1 = self.bus.last_stop_id
                    arr_t = estimate_arrival_time(bus.dep_t, stop0, stop1, self.time_dependent_travel_time, self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? bus behind is {curr_stop_idx-behind_last_stop_idx} stops behind')
                            print(f'behind bus departed {round((self.time - bus.dep_t)/60, 1)} '
                                  f'mins ago but is expected to run for {round((arr_t - bus.dep_t)/60, 1)}')
                        if arr_t - self.time >= 0:
                            expected_arr_t.append(arr_t)

        if expected_arr_t:
            return min(expected_arr_t) - self.time
        else:
            # this might happen if the episode is still running and we control the last dispatched trip
            print(f'no future arrival found at hour {self.time/60/60}')
            return SCHED_DEP_OUT[-1] - SCHED_DEP_OUT[-2]

    def record_trajectories(self, pickups=0, offs=0, denied_board=0, hold=0, skip=False):
        bus = self.bus
        trip_id = bus.active_trip[0].trip_id
        stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        try:
            scheduled_sec = bus.active_trip[0].schedule[stop_idx]
        except IndexError:
            print(bus.active_trip[0].trip_id)
            print(f'stop idx {stop_idx}')
            print(f'schedule length {len(bus.active_trip[0].schedule)}')
            print(f'schedule {bus.active_trip[0].schedule}')
            raise
        trajectory = [bus.last_stop_id, round(bus.arr_t, 1), round(bus.dep_t, 1),
                      len(bus.pax), pickups, offs, denied_board, hold, int(skip), scheduled_sec]
        self.trajectories_out[trip_id].append(trajectory)
        self.out_trip_record.append([trip_id, bus.last_stop_id, bus.arr_t, bus.dep_t, len(bus.pax), pickups, offs,
                                     denied_board, hold, int(skip), scheduled_sec, stop_idx+1])
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
            link_time_params_light = (link_time_params[0]*self.tt_factor, link_time_params[1], link_time_params[2])
            runtime = lognorm.rvs(*link_time_params_light)
            minim, maxim = link_time_extremes
            if runtime > maxim:
                runtime = min(EXTREME_TT_BOUND*maxim, runtime)
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
        self.focus_trips_finished = []
        self.bus = Bus(0, [])
        # for records
        self.out_trip_record = []
        self.in_trip_record = []
        self.trajectories_out = {}
        for trip_id in TRIP_IDS_OUT:
            self.trajectories_out[trip_id] = []
            self.trip_log.append(TripLog(trip_id, STOPS_OUTBOUND))
        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            trip_set = block_trip_set[1]
            self.buses.append(Bus(block_id, trip_set))
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            if trip.route_type == 0:
                bus.last_stop_id = STOPS_OUTBOUND[0]
            interval_delay = get_interval(trip.schedule[0], DELAY_INTERVAL_LENGTH_MINS) - DELAY_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            if trip.route_type == 0:
                delay = np.percentile(DEP_DELAY_DIST_OUT[interval_delay], rand_percentile)
            elif trip.route_type == 1:
                delay = np.percentile(DEP_DELAY1_DIST_IN[interval_delay], rand_percentile)
            else:
                assert trip.route_type == 2
                delay = np.percentile(DEP_DELAY2_DIST_IN[interval_delay], rand_percentile)
            # random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.schedule[0] + max(delay, 0)
            bus.dep_t = deepcopy(bus.next_event_time)
            bus.next_event_type = 3 if trip.route_type else 0
        # initialize passenger demand
        self.completed_pax = []
        self.completed_pax_record = []
        self.initialize_pax_demand()
        return False

    def initialize_pax_demand(self):
        pax_info = {}
        self.stops = []
        for orig_idx in range(len(STOPS_OUTBOUND)):
            self.stops.append(Stop(STOPS_OUTBOUND[orig_idx]))
            pax_info['arr_t'] = []
            pax_info['o_idx'] = []
            pax_info['d_idx'] = []
            for dest_idx in range(orig_idx + 1, len(STOPS_OUTBOUND)):
                for interval_idx in range(ODT_START_INTERVAL, ODT_END_INTERVAL):
                    start_edge_interval = interval_idx * ODT_INTERVAL_LEN_MIN * 60
                    end_edge_interval = start_edge_interval + ODT_INTERVAL_LEN_MIN * 60
                    odt_orig_idx = ODT_STOP_IDS.index(STOPS_OUTBOUND[orig_idx])
                    odt_dest_idx = ODT_STOP_IDS.index(STOPS_OUTBOUND[dest_idx])
                    od_rate = ODT_RATES_SCALED[interval_idx, odt_orig_idx, odt_dest_idx]
                    if od_rate > 0:
                        max_size = int(np.ceil(od_rate) * (ODT_INTERVAL_LEN_MIN / 60) * 10)
                        temp_pax_interarr_times = np.random.exponential(3600 / od_rate, size=max_size)
                        temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
                        if interval_idx == ODT_START_INTERVAL:
                            temp_pax_arr_times += PAX_INIT_TIME[orig_idx]
                        else:
                            temp_pax_arr_times += max(start_edge_interval, PAX_INIT_TIME[orig_idx])
                        temp_pax_arr_times = temp_pax_arr_times[
                            temp_pax_arr_times <= min(END_TIME_SEC, end_edge_interval)]
                        temp_pax_arr_times = temp_pax_arr_times.tolist()
                        if len(temp_pax_arr_times):
                            pax_info['arr_t'] += temp_pax_arr_times
                            pax_info['o_idx'] += [orig_idx] * len(temp_pax_arr_times)
                            pax_info['d_idx'] += [dest_idx] * len(temp_pax_arr_times)
            df = pd.DataFrame(pax_info).sort_values(by='arr_t')
            pax_sorted_info = df.to_dict('list')
            for o, d, at in zip(pax_sorted_info['o_idx'], pax_sorted_info['d_idx'],
                                pax_sorted_info['arr_t']):
                self.stops[orig_idx].pax.append(Passenger(o, d, at))
        return

    def fixed_stop_unload(self):
        # SWITCH POSITION AND ARRIVAL TIME
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.next_stop_id)
        bus.last_stop_id = STOPS_OUTBOUND[curr_stop_idx]
        bus.next_stop_id = STOPS_OUTBOUND[curr_stop_idx + 1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.arr_t
        self.stops[curr_stop_idx].last_arr_t.append(bus.arr_t)
        bus.offs = 0
        for p in self.bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                self.completed_pax.append(p)
                self.bus.active_trip[0].completed_pax.append(p)
                self.completed_pax_record.append([p.orig_idx, p.dest_idx, p.arr_time, p.board_time, p.alight_time,
                                                  p.trip_id, p.denied])
                self.bus.pax.remove(p)
                bus.offs += 1
        return

    def fixed_stop_load(self):
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        stops_served = bus.active_trip[0].stops
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME,
                             ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
        dwell_time = (bus.ons + bus.offs > 0) * dwell_time_pax
        if hold:
            dwell_time = max(hold, dwell_time_pax)
        bus.dep_t = self.time + dwell_time
        stops_served = bus.active_trip[0].stops
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= bus.dep_t and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.dep_t
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime

        last_stop_trip = bus.active_trip[0].stops[-1]
        bus.next_event_type = 2 if bus.next_stop_id == last_stop_trip else 1

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def outbound_ready_to_dispatch(self):
        bus = self.bus
        bus.last_stop_id = STOPS_OUTBOUND[0]
        bus.next_stop_id = STOPS_OUTBOUND[1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS_OUTBOUND[0]] = bus.arr_t
        self.stops[0].last_arr_t.append(bus.arr_t)
        return

    def outbound_dispatch(self, hold=0):
        bus = self.bus
        # trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        bus.denied = 0
        stops_served = bus.active_trip[0].stops
        for p in self.stops[0].pax.copy():
            if p.arr_time <= self.time and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        # FOR THOSE PAX WHO ARRIVE DURING THE DWELL TIME
        if bus.dep_t > self.time:
            for p in self.stops[0].pax.copy():
                if p.arr_time <= bus.dep_t and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS_OUTBOUND[0]] = bus.dep_t

        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=hold)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 1

        self.stops[0].passed_trips.append(bus.active_trip[0].trip_id)
        return

    def outbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        stops_served = bus.active_trip[0].stops
        last_stop_trip = stops_served[-1]
        self.trip_log[trip_idx].stop_arr_times[last_stop_trip] = bus.arr_t
        curr_stop_idx = STOPS_OUTBOUND.index(bus.next_stop_id)
        for p in self.bus.pax.copy():
            if p.dest_idx == curr_stop_idx:
                p.alight_time = float(self.time)
                # p.journey_time = float(p.alight_time - p.arr_time)
                self.completed_pax.append(p)
                self.completed_pax_record.append([p.orig_idx, p.dest_idx, p.arr_time, p.board_time, p.alight_time,
                                                  p.trip_id, p.denied])
                self.bus.active_trip[0].completed_pax.append(p)
                bus.pax.remove(p)
                bus.offs += 1
            else:
                print(f'what? pax destined for {p.dest_idx} and boarded in {p.orig_idx} which is stop {STOPS_OUTBOUND[p.dest_idx]} and only {len(stops_served)} stops served')
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = ACC_DEC_TIME + bus.offs * ALIGHTING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.offs > 0) * dwell_time

        assert len(bus.pax) == 0
        bus.last_stop_id = STOPS_OUTBOUND[curr_stop_idx]
        bus.dep_t = float(bus.arr_t) + dwell_time
        self.record_trajectories(offs=bus.offs)
        bus.finished_trips.append(bus.active_trip[0])
        if bus.active_trip[0].trip_id in FOCUS_TRIPS:
            self.focus_trips_finished.append(bus.active_trip[0].trip_id)
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            route_type = trip.route_type
            # types {1: long, 2: short}
            assert route_type == 1
            next_dep_time = max(bus.dep_t, trip.schedule[0])
            bus.next_event_time = next_dep_time
            bus.next_event_type = 3
        return

    def inbound_dispatch(self):
        bus = self.bus
        trip = bus.active_trip[0]
        trip_id = trip.trip_id
        route_type = trip.route_type
        if route_type == 1:
            interval_idx = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            run_time = np.percentile(TRIP_T1_DIST_IN[interval_idx], rand_percentile)
            arr_time = self.time + run_time
            start_stop_id = INBOUND_LONG_START_STOP
        else:
            assert route_type == 2
            interval_idx = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            run_time = np.percentile(TRIP_T2_DIST_IN[interval_idx], rand_percentile)
            arr_time = self.time + run_time
            start_stop_id = INBOUND_SHORT_START_STOP
        bus.dep_t = self.time
        schd_sec = bus.active_trip[0].schedule[0]
        self.trajectories_in[trip_id] = [[start_stop_id, bus.dep_t, schd_sec, 1]]
        self.in_trip_record.append([trip_id, start_stop_id, bus.dep_t, schd_sec, 1])
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        return

    def inbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_id = bus.active_trip[0].trip_id
        if bus.active_trip[0].stops[-1] == STOPS_OUTBOUND[0]:
            schd_sec = bus.active_trip[0].schedule[-1]
            self.trajectories_in[trip_id].append([STOPS_OUTBOUND[0], bus.arr_t, schd_sec])
            self.in_trip_record.append([trip_id, STOPS_OUTBOUND[0], bus.arr_t, schd_sec, len(bus.active_trip[0].stops)])
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            bus.last_stop_id = STOPS_OUTBOUND[0]
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            bus.next_event_time = max(self.time + layover, bus.active_trip[0].schedule[0])
            bus.next_event_type = 0
        return

    def next_event(self):
        active_buses = [bus for bus in self.buses if bus.active_trip]
        if active_buses:
            next_event_times = [bus.next_event_time for bus in active_buses]
            min_event_time = min(next_event_times)
            min_event_time_idxs = [i for i, x in enumerate(next_event_times) if x == min_event_time]
            assert min_event_time_idxs
            self.bus = active_buses[min_event_time_idxs[0]]
            self.time = float(self.bus.next_event_time)
            return False
        else:
            return True

    def prep(self):
        done = self.next_event()
        if done or all(elem in self.focus_trips_finished for elem in FOCUS_TRIPS):
            return True

        if self.bus.next_event_type == 0:
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            self.fixed_stop_unload()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.bus.next_event_type == 2:
            self.outbound_arrival()
            return False

        if self.bus.next_event_type == 3:
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 4:
            self.inbound_arrival()
            return False


class DetailedSimulationEnvWithControl(DetailedSimulationEnv):
    def __init__(self, control_strength=0.7, *args, **kwargs):
        super(DetailedSimulationEnvWithControl, self).__init__(*args, **kwargs)
        self.control_strength = control_strength
        self.hold_ts = np.array([])

    def decide_bus_holding(self):
        # CHANGE------------
        # IT SHOULD BE HOLD TIME = MAX(0, MIN(PREV_ARR_T + LIMIT_HOLD - SELF.TIME, (BW_H-FW_H)/2 - SELF.TIME)
        # for previous trip
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(self.bus.last_stop_id)
        stop = self.stops[curr_stop_idx]
        backward_headway = self.backward_headway()
        last_arr_t = self.stops[curr_stop_idx].last_arr_t
        if last_arr_t:
            forward_headway = self.time - last_arr_t[-2]
        else:
            print(f'what? last arrival time shows nan for controlled trip at time {self.time/60/60}')
            forward_headway = SCHED_DEP_OUT[1] - SCHED_DEP_OUT[0]
        if stop.stop_id == STOPS_OUTBOUND[0]:
            holding_time = min(LIMIT_HOLDING, max((backward_headway - forward_headway) / 2, 0))
            if self.hold_adj_factor > 0.0 and holding_time > 0:
                holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
            assert holding_time >= 0
            if self.bus.active_trip[0].trip_id in FOCUS_TRIPS:
                self.hold_ts = np.append(self.hold_ts, holding_time)
            self.outbound_dispatch(hold=holding_time)
        else:
            self.fixed_stop_load()
            holding_time = min(LIMIT_HOLDING, max((backward_headway - forward_headway)/2,0))
            if self.hold_adj_factor > 0.0 and holding_time > 0:
                holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
            assert holding_time >= 0
            if self.bus.active_trip[0].trip_id in FOCUS_TRIPS:
                self.hold_ts = np.append(self.hold_ts, holding_time)
            self.fixed_stop_depart(hold=holding_time)
        return

    def prep(self):
        done = self.next_event()
        if done or all(elem in self.focus_trips_finished for elem in FOCUS_TRIPS):
            return True

        if self.bus.next_event_type == 0:
            arrival_stop = STOPS_OUTBOUND[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.outbound_ready_to_dispatch()
                    self.decide_bus_holding()
                    return False
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
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
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.bus.next_event_type == 2:
            self.outbound_arrival()
            return False

        if self.bus.next_event_type == 3:
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 4:
            self.inbound_arrival()
            return False


class DetailedSimulationEnvWithDeepRL(DetailedSimulationEnv):
    def __init__(self, estimate_pax=False,  weight_ride_t=0.0, weight_cv_hw=0.94, *args, **kwargs):
        super(DetailedSimulationEnvWithDeepRL, self).__init__(*args, **kwargs)
        self.trips_sars = {}
        self.bool_terminal_state = False
        self.pool_sars = []
        self.estimate_pax = estimate_pax
        self.weight_ride_t = weight_ride_t
        self.weight_cv_hw = weight_cv_hw

    def reset_simulation(self):
        self.time = START_TIME_SEC
        self.focus_trips_finished = []
        self.bool_terminal_state = False
        self.pool_sars = []
        self.bus = Bus(0, [])

        # for records
        self.out_trip_record = []
        self.in_trip_record = []
        self.trajectories_out = {}
        self.trips_sars = {}
        for trip_id in TRIP_IDS_OUT:
            self.trajectories_out[trip_id] = []
            self.trip_log.append(TripLog(trip_id, STOPS_OUTBOUND))
        for trip_id in CONTROL_TRIP_IDS:
            self.trips_sars[trip_id] = []

        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            trip_set = block_trip_set[1]
            self.buses.append(Bus(block_id, trip_set))
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            trip = bus.active_trip[0]
            if trip.route_type == 0:
                bus.last_stop_id = STOPS_OUTBOUND[0]

            interval_delay = get_interval(trip.schedule[0], DELAY_INTERVAL_LENGTH_MINS) - DELAY_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            if trip.route_type == 0:
                dep_delay = np.percentile(DEP_DELAY_DIST_OUT[interval_delay], rand_percentile)
            elif trip.route_type == 1:
                dep_delay = np.percentile(DEP_DELAY1_DIST_IN[interval_delay], rand_percentile)
            else:
                assert trip.route_type == 2
                dep_delay = np.percentile(DEP_DELAY2_DIST_IN[interval_delay], rand_percentile)
            # random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.schedule[0] + max(dep_delay, 0)
            bus.dep_t = deepcopy(bus.next_event_time)
            bus.next_event_type = 3 if trip.route_type else 0
        # initialize passenger demand
        self.completed_pax = []
        self.completed_pax_record = []
        self.initialize_pax_demand()
        return False

    def fixed_stop_load(self, skip=False):
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        stops_served = bus.active_trip[0].stops
        for p in self.stops[curr_stop_idx].pax.copy():
            if p.arr_time <= self.time and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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
        trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
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
        stops_served = bus.active_trip[0].stops
        if not skip:
            for p in self.stops[curr_stop_idx].pax.copy():
                if p.arr_time <= bus.dep_t and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        self.trip_log[trip_idx].stop_dep_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.dep_t
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime

        last_stop = bus.active_trip[0].stops[-1]
        bus.next_event_type = 2 if bus.next_stop_id == last_stop else 1
        if TRIP_IDS_OUT[trip_idx] == 911880020 and curr_stop_idx > 60:
            print(f'for last stop id {last_stop} next stop is {bus.next_stop_id} '
                  f'with index {curr_stop_idx} therefore next event type is {bus.next_event_type}')
        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        last_arr = self.stops[curr_stop_idx].last_arr_t[-2]
        second_last_arr = self.stops[curr_stop_idx].last_arr_t[-3]
        forward_headway = self.time - last_arr
        backward_headway = self.backward_headway()
        trip_id = self.bus.active_trip[0].trip_id
        trip_sars = self.trips_sars[trip_id]
        bus_load = len(self.bus.pax)

        route_progress = curr_stop_idx/len(STOPS_OUTBOUND)
        if self.estimate_pax:
            interval_idx = get_interval(self.time, ODT_INTERVAL_LEN_MIN)
            odt_stop_idx = ODT_STOP_IDS.index(bus.last_stop_id)
            arr_rate = SCALED_ARR_RATES[interval_idx, odt_stop_idx]
            pax_at_stop = round(forward_headway * (arr_rate / 3600))
        else:
            pax_at_stop = 0
            for p in self.stops[curr_stop_idx].pax.copy():
                if p.arr_time <= self.time:
                    pax_at_stop += 1
                else:
                    break
        prev_fw_h = last_arr - second_last_arr
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
            hold_time = (action - 1) * BASE_HOLDING_TIME
            if self.hold_adj_factor > 0.0 and hold_time > 0:
                hold_time = np.random.uniform(self.hold_adj_factor * hold_time, hold_time)
                assert hold_time >= 0
            if bus.last_stop_id == STOPS_OUTBOUND[0]:
                self.outbound_dispatch(hold=hold_time)
            else:
                self.fixed_stop_load()
                self.fixed_stop_depart(hold=hold_time)
        else:
            if bus.last_stop_id == STOPS_OUTBOUND[0]:
                self.outbound_dispatch()
            else:
                self.fixed_stop_load(skip=True)
                self.fixed_stop_depart(skip=True)
        return

    def delayed_reward(self, s0, s1, neighbor_prev_hold):
        bus = self.bus
        s0_idx = STOPS_OUTBOUND.index(s0) # you check from those pax boarding next to the control stop (impacted)
        s1_idx = STOPS_OUTBOUND.index(s1) # next control point
        trip_id = bus.active_trip[0].trip_id
        trip_idx = TRIP_IDS_OUT.index(trip_id)

        # FRONT NEIGHBOR'S REWARD
        agent_trip_idx = trip_idx - 1
        agent_trip_id = TRIP_IDS_OUT[agent_trip_idx]
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

        t0_agent = self.trip_log[agent_trip_idx].stop_arr_times[s0]
        t1_agent = self.trip_log[agent_trip_idx].stop_arr_times[s1]

        t0_behind = self.trip_log[trip_idx].stop_arr_times[s0]
        t1_behind = self.trip_log[trip_idx].stop_arr_times[s1]

        for pax in self.completed_pax:
            if pax.trip_id == agent_trip_id and pax.orig_idx in stop_idx_set:
                pax_count += 1
                wait = pax.board_time - pax.arr_time
                ride = min(pax.alight_time - pax.board_time, t1_agent - pax.board_time)
                assert ride > 0
                if pax.orig_idx == stop_idx_set[0] and wait != 0.0:
                    wait += neighbor_prev_hold
                sum_rew_agent_wait_time += wait
                sum_rew_agent_ride_time += ride
                agent_board_count += 1
            elif pax.trip_id == agent_trip_id and pax.dest_idx in stop_idx_set[1:]:
                pax_count += 1
                ride = pax.alight_time - t0_agent
                assert ride > 0
                sum_rew_agent_ride_time += ride
        active_buses = [bus for bus in self.buses if bus.active_trip]
        front_active_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == agent_trip_id]
        if front_active_bus:
            for pax in front_active_bus[0].pax:
                if pax.orig_idx in stop_idx_set:
                    pax_count += 1
                    wait = pax.board_time - pax.arr_time
                    ride = t1_agent - pax.board_time
                    assert ride > 0
                    if pax.orig_idx == stop_idx_set[0] and wait != 0.0:
                        wait += neighbor_prev_hold
                    sum_rew_agent_wait_time += wait
                    sum_rew_agent_ride_time += ride
                    agent_board_count += 1
                elif pax.orig_idx in stop_idx_before_set:
                    pax_count += 1
                    ride = t1_agent - t0_agent
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
                wait = pax.board_time - pax.arr_time
                sum_rew_behind_wait_time += wait
                ride = t1_behind - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.orig_idx in stop_idx_before_set:
                pax_count += 1
                ride = t1_behind - t0_behind

                assert ride > 0
                sum_rew_behind_ride_time += ride
        for pax in self.completed_pax:
            if pax.trip_id == trip_id and pax.orig_idx in stop_idx_set:
                pax_count += 1
                wait = pax.board_time - pax.arr_time
                sum_rew_behind_wait_time += wait
                ride = pax.alight_time - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.trip_id == trip_id and pax.dest_idx in stop_idx_set[1:]:
                pax_count += 1
                ride = pax.alight_time - t0_behind
                assert ride > 0
                sum_rew_behind_ride_time += ride
        reward_wait = (sum_rew_behind_wait_time + sum_rew_agent_wait_time)
        reward_ride = sum_rew_agent_ride_time
        reward = - (reward_wait + self.weight_ride_t * reward_ride) / 60 / 60
        return reward

    def update_rewards(self, simple_reward=False):
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
            trip_idx = TRIP_IDS_OUT.index(trip_id)
            neighbor_trip_id = TRIP_IDS_OUT[trip_idx - 1]
            sars_idx = k - 1
            assert sars_idx >= 0
            assert sars_idx < len(self.trips_sars[neighbor_trip_id])
            assert sars_idx < len(self.trips_sars[trip_id])
            if simple_reward:
                weight_cv_hw = self.weight_cv_hw
                prev_sars = self.trips_sars[trip_id][sars_idx]
                prev_action = prev_sars[1]
                prev_hold = (prev_action - 1) * BASE_HOLDING_TIME if prev_action > 1 else 0
                fw_h1 = prev_sars[3][IDX_FW_H]
                planned_fw_h = SCHED_DEP_OUT[trip_idx] - SCHED_DEP_OUT[trip_idx - 1]
                hw_variation = (fw_h1 - planned_fw_h)/planned_fw_h
                if self.hold_adj_factor > 0.0:
                    prev_hold = prev_hold - (prev_hold - self.hold_adj_factor * prev_hold)/2
                    reward = - weight_cv_hw*hw_variation * hw_variation - (1-weight_cv_hw)*(prev_hold/LIMIT_HOLDING)
                else:
                    reward = - weight_cv_hw*hw_variation * hw_variation - (1-weight_cv_hw)*(prev_hold/LIMIT_HOLDING)
                self.trips_sars[trip_id][sars_idx][2] = reward
                self.pool_sars.append(self.trips_sars[trip_id][sars_idx] + [self.bool_terminal_state])
            else:
                prev_sars = self.trips_sars[neighbor_trip_id][sars_idx]
                prev_action = prev_sars[1]
                prev_hold = (prev_action - 1) * BASE_HOLDING_TIME if prev_action > 1 else 0
                reward = self.delayed_reward(prev_control_stop, curr_control_stop, prev_hold)
                assert self.trips_sars[neighbor_trip_id][sars_idx][3]
                self.trips_sars[neighbor_trip_id][sars_idx][2] = reward
                self.pool_sars.append(self.trips_sars[neighbor_trip_id][sars_idx] + [self.bool_terminal_state])
        return

    def prep(self):
        done = self.next_event()
        if done or all(elem in self.focus_trips_finished for elem in FOCUS_TRIPS):
            return True
        if self.bus.next_event_type == 0:
            arrival_stop = STOPS_OUTBOUND[0]
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS:
                if arrival_stop in CONTROLLED_STOPS[:-1]:
                    self.bool_terminal_state = False
                    self.outbound_ready_to_dispatch()
                    self._add_observations()
                    return False
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
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
            self.outbound_arrival()
            return self.prep()

        if self.bus.next_event_type == 3:
            self.inbound_dispatch()
            return self.prep()

        if self.bus.next_event_type == 4:
            self.inbound_arrival()
            return self.prep()


class DetailedSimulationEnvWithDispatching(DetailedSimulationEnv):
    def __init__(self, missing_trips=False, *args, **kwargs):
        super(DetailedSimulationEnvWithDispatching, self).__init__(*args, **kwargs)
        self.missing_trips = missing_trips

    def reset_simulation(self):
        self.time = START_TIME_SEC
        self.focus_trips_finished = []
        self.bus = Bus(0, [])
        # for records
        self.out_trip_record = []
        self.in_trip_record = []
        self.trajectories_out = {}
        for trip_id in TRIP_IDS_OUT:
            self.trajectories_out[trip_id] = []
            self.trip_log.append(TripLog(trip_id, STOPS_OUTBOUND))
        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            trip_set = block_trip_set[1]
            self.buses.append(Bus(block_id, trip_set))
        # insert here function for randomly assigning cancelled buses (blocks)
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            if random.uniform(0, 1) <= PROB_CANCELLED_BLOCK and self.missing_trips:
                bus.cancelled = True
                # bus.will_cancel = True
                # bus.next_event_time = max(START_TIME_SEC, trip.schedule[0] - CANCEL_NOTICE_TIME)
                # bus.next_event_type = 6 # cancellation event
            else:
                bus.active_trip.append(bus.pending_trips[0])
                bus.pending_trips.pop(0)
                trip = bus.active_trip[0]
                if trip.route_type == 0:
                    bus.last_stop_id = STOPS_OUTBOUND[0]
                interval_delay = get_interval(trip.schedule[0], DELAY_INTERVAL_LENGTH_MINS) - DELAY_START_INTERVAL
                rand_percentile = np.random.uniform(0.0, 100.0)
                if trip.route_type == 0:
                    delay = np.percentile(DEP_DELAY_DIST_OUT[interval_delay], rand_percentile)
                elif trip.route_type == 1:
                    delay = np.percentile(DEP_DELAY1_DIST_IN[interval_delay], rand_percentile)
                else:
                    assert trip.route_type == 2
                    delay = np.percentile(DEP_DELAY2_DIST_IN[interval_delay], rand_percentile)
                bus.next_event_time = trip.schedule[0] + max(delay, 0)
                bus.dep_t = deepcopy(bus.next_event_time)
                bus.next_event_type = 3 if trip.route_type else 0
        # initialize passenger demand
        self.completed_pax = []
        self.completed_pax_record = []
        self.initialize_pax_demand()
        return False

    def inbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_id = bus.active_trip[0].trip_id
        if bus.active_trip[0].stops[-1] == STOPS_OUTBOUND[0]:
            schd_sec = bus.active_trip[0].schedule[-1]
            self.trajectories_in[trip_id].append([STOPS_OUTBOUND[0], bus.arr_t, schd_sec])
            self.in_trip_record.append([trip_id, STOPS_OUTBOUND[0], bus.arr_t, schd_sec, len(bus.active_trip[0].stops)])
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            bus.last_stop_id = STOPS_OUTBOUND[0]
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            # NEXT EVENT IS THE EARLIEST BETWEEN THE READY TIME AND THE EARLY DEPARTURE LIMIT
            ready_time = self.time + layover
            bus.next_event_time = max(ready_time, bus.active_trip[0].schedule[0] - EARLY_DEP_LIMIT_SEC)
            bus.next_event_type = 5
        return

    def actual_future_headways(self):
        # terminal
        # search for active buses inbound and compute the time they will pick up pax at terminal (arrival time)
        future_dep_t = []
        # this will collect the expected arrival time of all trips departing
        for bus in self.buses:
            if not bus.cancelled:
                # HERE WE TAKE TRIPS THAT ARE MID-ROUTE IN THE INBOUND DIRECTION
                if bus.active_trip and bus.active_trip[0].route_type in [1, 2] and bus.pending_trips and bus.bus_id != self.bus.bus_id:
                    assert bus.next_event_type == 4 or bus.next_event_type == 3
                    dep_sched_dev = bus.dep_t - bus.active_trip[0].schedule[0]
                    # interv_idx = get_interval(prev_dep_t, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL

                    # if bus.active_trip[0].route_type == 1:
                    #     estimated_in_run_t = np.mean(TRIP_T1_DIST_IN[interv_idx])
                    # else:
                    #     estimated_in_run_t = np.mean(TRIP_T2_DIST_IN[interv_idx])
                    ready_time = bus.active_trip[0].schedule[-1] + dep_sched_dev  + MIN_LAYOVER_T
                    next_sched_dep_t = bus.pending_trips[0].schedule[0]
                    if self.bus.last_stop_id == STOPS_OUTBOUND[0]:
                        future_dep_t.append(max(ready_time, next_sched_dep_t))

                # HERE WE TAKE TRIPS THAT ARE IN THE TERMINAL WAITING TO BE DISPATCHED
                if bus.active_trip and bus.active_trip[0].route_type == 0 and bus.next_event_type == 0 and bus.bus_id != self.bus.bus_id:
                    future_dep_t.append(bus.next_event_time)
            if len(future_dep_t) >= FUTURE_HW_HORIZON:
                future_dep = sorted(future_dep_t)[:FUTURE_HW_HORIZON]
                future_dep.insert(0, self.time)
                future_dep_hw = [future_dep[i] - future_dep[i-1] for i in range(1, len(future_dep))]
                return future_dep_hw
            else:
                # this might happen if the episode is still running and we control the last dispatched trip
                print(f'no future arrival found at hour {self.time/60/60}')
                return [SCHED_DEP_OUT[-1] - SCHED_DEP_OUT[-2]]*FUTURE_HW_HORIZON

    def report_observations(self):
        bus = self.bus
        sched_dep = bus.active_trip[0].schedule[0]

        ref_dep_t = max(self.time, sched_dep) # this will be the reference point - future and past hw extracted based on
        past_sched_dep = np.sort(sched_deps_arr[sched_deps_arr < ref_dep_t])[-PAST_HW_HORIZON:].tolist()
        past_sched_dep.append(self.time)
        past_sched_hw = [past_sched_dep[i] - past_sched_dep[i-1] for i in range(1, len(past_sched_dep))]

        past_actual_dep = self.stops[0].last_dep_t[-PAST_HW_HORIZON:]
        past_actual_dep.append(self.time)
        past_actual_hw = [past_actual_dep[i] - past_actual_dep[i-1] for i in range(1, len(past_actual_dep))]

        future_sched_dep = np.sort(sched_deps_arr[sched_deps_arr > ref_dep_t])[:FUTURE_HW_HORIZON].tolist()
        future_sched_dep.insert(0, self.time)
        future_sched_hw = [future_sched_dep[i] - future_sched_dep[i-1] for i in range(1, len(future_sched_dep))]
        
        future_actual_hw = self.actual_future_headways()

        sched_dev = self.time - sched_dep

        bus.next_event_time = ref_dep_t
        print(f'new event -----')
        print(f'current time is {bus.next_event_time} and next event time is {bus.next_event_time}')
        print(f'time with respect to schedule for trip {bus.active_trip[0].trip_id} is {self.time - bus.active_trip[0].schedule[0]}')

        bus.next_event_type = 0
        return

    def outbound_ready_to_dispatch(self):
        bus = self.bus
        bus.last_stop_id = STOPS_OUTBOUND[0]
        bus.next_stop_id = STOPS_OUTBOUND[1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS_OUTBOUND[0]] = bus.arr_t
        self.stops[0].last_arr_t.append(bus.arr_t)
        return

    def outbound_dispatch(self, hold=0):
        bus = self.bus
        bus.denied = 0
        stops_served = bus.active_trip[0].stops
        for p in self.stops[0].pax.copy():
            if p.arr_time <= self.time and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        # FOR THOSE PAX WHO ARRIVE DURING THE DWELL TIME
        if bus.dep_t > self.time:
            for p in self.stops[0].pax.copy():
                if p.arr_time <= bus.dep_t and STOPS_OUTBOUND[p.dest_idx] in stops_served:
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

        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS_OUTBOUND[0]] = bus.dep_t

        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=hold)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 1

        self.stops[0].passed_trips.append(bus.active_trip[0].trip_id)
        self.stops[0].last_dep_t.append(bus.dep_t)
        return

    def prep(self):
        done = self.next_event()
        if done or all(elem in self.focus_trips_finished for elem in FOCUS_TRIPS):
            return True

        if self.bus.next_event_type == 0:
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            self.fixed_stop_unload()
            self.fixed_stop_load()
            self.fixed_stop_depart()
            return False

        if self.bus.next_event_type == 2:
            self.outbound_arrival()
            return False

        if self.bus.next_event_type == 3:
            self.inbound_dispatch()
            return False

        if self.bus.next_event_type == 4:
            self.inbound_arrival()
            return False

        if self.bus.next_event_type == 5: # outbound terminal dispatch decision point
            self.report_observations()
            return
