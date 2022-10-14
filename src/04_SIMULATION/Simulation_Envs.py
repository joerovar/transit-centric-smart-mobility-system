from Variable_Inputs import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from Simulation_Classes import Passenger, Stop, Bus, TripLog


def get_interval(t, interval_length):
    # time ins seconds, interval length ins minutes
    interval = int(t / (interval_length * 60))
    return interval

def time_string(secs):
    return str(timedelta(seconds=round(secs)))

def _compute_reward(action, fw_h, bw_h, prev_bw_h, prev_fw_h, prev_pax_at_s):
    hw_diff0 = abs(prev_fw_h - prev_bw_h)
    hw_diff1 = abs(fw_h - bw_h)
    reward = hw_diff0 - hw_diff1
    if prev_pax_at_s and action == SKIP_ACTION:
        reward -= 0.5 * prev_bw_h
    return reward


def even_hw_decision(bw_h, fw_h, hold_max):
    return min(max((bw_h - fw_h) / 2, 0), hold_max)


def single_hw_decision(fw_h, sched_fw_h, hold_max):
    return min(max((sched_fw_h - fw_h) / 2, 0), hold_max)


def get_reward(fw_hs, bw_hs, hold_t, weight_hold_t):
    reward = 0
    for hs in (fw_hs, bw_hs):
        reward -= np.power((hs[1] - hs[0]) / hs[1], 2)
    reward -= weight_hold_t * hold_t
    return reward


def load_pax(stop, t, bus, bus_cap, next_stops, trip_id, on_dwell=False):
    pax = sorted(stop.pax, key=lambda x: x.arr_time)
    if len(pax) > 1:
        sorted_times = [p.arr_time for p in pax]
        if sorted_times[-1] > t:
            idx = next(x[0] for x in enumerate(sorted_times) if x[1] > t)
            curr_pax = pax[:idx]
            later_pax = pax[idx:]
        else:
            curr_pax = pax
            later_pax = []
    elif len(pax) == 1:
        if pax[0].arr_time <= t:
            curr_pax = pax
            later_pax = []
        else:
            curr_pax = []
            later_pax = pax
    else:
        return
    valid_pax = [p for p in curr_pax if str(p.dest_id) in next_stops]
    non_valid_pax = [p for p in curr_pax if str(p.dest_id) not in next_stops]
    later_pax = non_valid_pax + later_pax
    max_pax = bus_cap - len(bus.pax)
    if len(valid_pax) <= max_pax:
        board_pax = valid_pax
        denied_pax = []
    else:
        board_pax = valid_pax[:max_pax]
        denied_pax = valid_pax[max_pax:]
    for p in board_pax:
        p.board_time = float(t)
        p.trip_id = trip_id
    for p in denied_pax:
        p.denied = 1
    stop.pax = denied_pax + later_pax
    bus.pax += board_pax
    if on_dwell:
        bus.ons += len(board_pax)
    else:
        bus.ons = len(board_pax)
    bus.denied = len(denied_pax)
    return


class SimulationEnv:
    def __init__(self, tt_factor=1.0, hold_adj_factor=0.0, bus_capacity=HIGH_CAPACITY):
        self.stops = {}
        self.completed_pax = []
        self.buses = []
        self.bus = Bus(0, [])
        self.trip_log = []
        self.focus_trips_finished = []
        self.completed_pax_record = []
        self.out_trip_record = []
        self.in_trip_record = []
        self.time = 0.0
        self.tt_factor = tt_factor
        self.hold_adj_factor = hold_adj_factor
        self.bus_capacity = bus_capacity

    def estimate_arrival_time(self, start_time, start_stop, end_stop, current_time):
        stops = self.bus.active_trip[0].stops
        temp_arr_t = start_time
        start_stop_idx = stops.index(start_stop)
        end_stop_idx = stops.index(end_stop)
        for i in range(start_stop_idx, end_stop_idx):
            stop0 = stops[i]
            stop1 = stops[i + 1]
            interv = get_interval(temp_arr_t, TIME_BIN_MINS)
            if temp_arr_t > END_TIME_SEC:
                return np.nan
            interv_idx = interv - TIME_START_INTERVAL
            mean_runtime = LINK_TIMES_MEAN[stop0 + '-' + stop1][interv_idx]
            temp_arr_t = max(temp_arr_t + mean_runtime, current_time)
            assert mean_runtime
        arrival_time = temp_arr_t
        return arrival_time

    def backward_headway(self):
        # terminal
        # search for active buses inbound and compute the time they will pick up pax at terminal (arrival time)
        expected_arr_t = []
        # this will collect the expected arrival time of all trips departing
        for bus in self.buses:
            # HERE WE TAKE TRIPS THAT ARE MID-ROUTE IN THE INBOUND DIRECTION
            if bus.active_trip and bus.active_trip[
                0].direction == 1 and bus.pending_trips and bus.bus_id != self.bus.bus_id:
                assert bus.next_event_type == 4
                prev_dep_t = bus.dep_t
                interv_idx = get_interval(prev_dep_t, TRIP_TIME_BIN_MINS) - TRIP_TIME_START_INTERVAL
                stops = bus.active_trip[0].stops
                estimated_in_run_t = np.mean(RUN_T_DISTR_IN[stops[0] + '-' + stops[-1]][interv_idx])
                ready_time = prev_dep_t + estimated_in_run_t + MIN_LAYOVER_T
                next_sched_t = bus.pending_trips[0].schedule[0]
                if self.bus.last_stop_id == STOPS_OUT_FULL_PATT[0]:
                    expected_arr_t.append(max(ready_time, next_sched_t))
                else:
                    terminal_dep_t = max(ready_time, next_sched_t)
                    arr_t = self.estimate_arrival_time(terminal_dep_t, STOPS_OUT_FULL_PATT[0], self.bus.last_stop_id,
                                                       self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? on buses on inbound')
                        expected_arr_t.append(arr_t)

            # HERE WE TAKE TRIPS THAT ARE IN THE TERMINAL WAITING TO BE DISPATCHED
            if not bus.active_trip and bus.pending_trips and bus.pending_trips[
                0].direction == 0 and bus.bus_id != self.bus.bus_id:
                if self.bus.last_stop_id == STOPS_OUT_FULL_PATT[0]:
                    expected_arr_t.append(max(bus.next_event_time, bus.pending_trips[0].schedule[0]))
                else:
                    terminal_dep_t = max(bus.next_event_time, bus.pending_trips[0].schedule[0])
                    arr_t = self.estimate_arrival_time(terminal_dep_t, STOPS_OUT_FULL_PATT[0], self.bus.last_stop_id,
                                                       self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? on garaged buses')
                        expected_arr_t.append(arr_t)

            # only for trips that are at a midpoint control route should check buses behind ins the outbound direction

            if self.bus.last_stop_id != STOPS_OUT_FULL_PATT[0] and bus.active_trip and bus.active_trip[
                0].direction == 0 and bus.next_event_type == 1:
                stops = bus.active_trip[0].stops
                behind_last_stop_idx = stops.index(bus.last_stop_id)
                curr_stop_idx = stops.index(self.bus.last_stop_id)
                if behind_last_stop_idx < curr_stop_idx:
                    stop0 = bus.last_stop_id
                    stop1 = self.bus.last_stop_id
                    arr_t = self.estimate_arrival_time(bus.dep_t, stop0, stop1, self.time)
                    if not np.isnan(arr_t):
                        if arr_t - self.time < 0:
                            print(f'what? bus behind is {curr_stop_idx - behind_last_stop_idx} stops behind')
                            print(f'behind bus departed {round((self.time - bus.dep_t) / 60, 1)} '
                                  f'mins ago but is expected to run for {round((arr_t - bus.dep_t) / 60, 1)}')
                        if arr_t - self.time >= 0:
                            expected_arr_t.append(arr_t)

        if expected_arr_t:
            return min(expected_arr_t) - self.time
        else:
            # this might happen if the episode is still running and we control the last dispatched trip
            print(f'no future arrival found at hour {self.time / 60 / 60}')
            return None

    def record_trajectories(self, pickups=0, offs=0, denied_board=0, hold=0, skip=False):
        bus = self.bus
        trip_id = bus.active_trip[0].trip_id
        stops = bus.active_trip[0].stops
        stop_idx = stops.index(bus.last_stop_id)
        try:
            scheduled_sec = bus.active_trip[0].schedule[stop_idx]
            dist_traveled = bus.active_trip[0].dist_traveled[stop_idx]
        except IndexError:
            print(bus.active_trip[0].trip_id)
            print(f'stop idx {stop_idx}')
            print(f'schedule length {len(bus.active_trip[0].schedule)}')
            print(f'schedule {bus.active_trip[0].schedule}')
            raise
        self.out_trip_record.append([bus.bus_id, trip_id, bus.last_stop_id, bus.arr_t, bus.dep_t,
                                     len(bus.pax), pickups, offs, denied_board, hold, int(skip),
                                     scheduled_sec, stop_idx + 1, dist_traveled, int(bus.expressed)])
        return

    def get_travel_time(self, stop0, stop1):
        bus = self.bus
        link = str(stop0) + '-' + str(stop1)
        stops = bus.active_trip[0].stops
        stop_idx = stops.index(bus.last_stop_id)
        interv = get_interval(self.time, TIME_BIN_MINS)
        interv_idx = interv - TIME_START_INTERVAL
        last_link = stops[-2] + '-' + stops[-1]
        first_link = stops[0] + '-' + stops[1]
        if link in [first_link, last_link]:
            # problematic
            return (bus.active_trip[0].schedule[stop_idx + 1] - bus.active_trip[0].schedule[stop_idx]) * 1.8
        link_time_params = LINK_TIMES_PARAMS[link][interv_idx]
        link_time_extremes = LINK_TIMES_EXTREMES[link][interv_idx]
        if type(link_time_params) is not tuple and np.isnan(link_time_params):
            return bus.active_trip[0].schedule[stop_idx + 1] - bus.active_trip[0].schedule[stop_idx]
        try:
            link_time_params_light = (link_time_params[0] * self.tt_factor, link_time_params[1], link_time_params[2])
        except TypeError:
            print(link_time_params)
            raise
        runtime = lognorm.rvs(*link_time_params_light)
        minim, maxim = link_time_extremes
        if runtime > maxim:
            runtime = min(EXTREME_TT_BOUND * maxim, runtime)
            runtime = max(EXTREME_TT_BOUND * minim, runtime)
        return runtime

    def reset_simulation(self):
        self.time = START_TIME_SEC
        self.focus_trips_finished = []
        self.bus = Bus(0, [])
        # for records
        self.out_trip_record = []
        self.in_trip_record = []

        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            trip_set = block_trip_set[1]
            self.buses.append(Bus(block_id, trip_set))
        # initialize passenger demand
        self.completed_pax = []
        self.completed_pax_record = []
        self.initialize_pax_demand()
        # initialize bus trips (the first trip for each bus)
        for bus in self.buses:
            for trip in bus.pending_trips:
                self.trip_log.append(TripLog(trip.trip_id, trip.stops))
            trip = bus.pending_trips[0]
            interval_idx = get_interval(trip.schedule[0], DELAY_BIN_MINS) - DELAY_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            if trip.direction == 0:
                delay_dist = DELAY_DISTR_OUT[interval_idx]
            else:
                delay_dist = DELAY_DISTR_IN[trip.stops[0]][interval_idx]  # DICTIONARY TYPE
            delay = np.percentile(delay_dist, rand_percentile)
            bus.next_event_time = trip.schedule[0] + max(delay, 0)
            bus.next_event_type = 3 if trip.direction else 0
        return False

    def initialize_pax_demand(self):
        pax_info = {}
        self.stops = {}
        for orig in STOPS_OUT_ALL:
            self.stops[orig] = Stop(orig, STOPS_OUT_INFO[orig])
            pax_info['arr_t'] = []
            pax_info['o_id'] = []
            pax_info['d_id'] = []
            odt_orig_idx = ODT_STOP_IDS.index(orig)
            for dest in [stop for stop in STOPS_OUT_ALL if stop != orig]:
                odt_dest_idx = ODT_STOP_IDS.index(dest)
                for interval_idx in range(ODT_START_INTERVAL, ODT_END_INTERVAL):
                    start_edge_interval = interval_idx * ODT_BIN_MINS * 60
                    end_edge_interval = start_edge_interval + ODT_BIN_MINS * 60
                    od_rate = ODT_FLOWS[interval_idx, odt_orig_idx, odt_dest_idx]
                    if od_rate > 0:
                        max_size = int(np.ceil(od_rate) * (ODT_BIN_MINS / 60) * 10)
                        temp_pax_interarr_times = np.random.exponential(3600 / od_rate, size=max_size)
                        temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
                        sched_at_stop = self.stops[orig].sched_t
                        initial_t = sched_at_stop[0] - (sched_at_stop[1] - sched_at_stop[0])
                        if interval_idx == ODT_START_INTERVAL:
                            temp_pax_arr_times += initial_t
                        else:
                            temp_pax_arr_times += max(start_edge_interval, initial_t)
                        temp_pax_arr_times = temp_pax_arr_times[
                            temp_pax_arr_times <= min(END_TIME_SEC, end_edge_interval)]
                        temp_pax_arr_times = temp_pax_arr_times.tolist()
                        if len(temp_pax_arr_times):
                            pax_info['arr_t'] += temp_pax_arr_times
                            pax_info['o_id'] += [orig] * len(temp_pax_arr_times)
                            pax_info['d_id'] += [dest] * len(temp_pax_arr_times)
            df = pd.DataFrame(pax_info).sort_values(by='arr_t')
            pax_sorted_info = df.to_dict('list')
            for o, d, at in zip(pax_sorted_info['o_id'], pax_sorted_info['d_id'], pax_sorted_info['arr_t']):
                self.stops[orig].pax.append(Passenger(o, d, at))
        return

    def fixed_stop_unload(self):
        # SWITCH POSITION AND ARRIVAL TIME
        bus = self.bus
        stops = bus.active_trip[0].stops
        curr_stop_idx = stops.index(bus.next_stop_id)
        bus.last_stop_id = stops[curr_stop_idx]
        bus.next_stop_id = stops[curr_stop_idx + 1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[bus.last_stop_id] = bus.arr_t
        self.stops[bus.last_stop_id].last_arr_t.append(bus.arr_t)
        bus.offs = 0
        for p in self.bus.pax.copy():
            if p.dest_id == bus.last_stop_id:
                p.alight_time = float(self.time)
                self.completed_pax.append(p)
                self.bus.active_trip[0].completed_pax.append(p)
                self.completed_pax_record.append([p.orig_id, p.dest_id, p.arr_time, p.board_time, p.alight_time,
                                                  p.trip_id, p.denied])
                self.bus.pax.remove(p)
                bus.offs += 1
        return

    def fixed_stop_load(self):
        bus = self.bus
        stops = bus.active_trip[0].stops
        curr_stop_idx = stops.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        next_stops = stops[curr_stop_idx + 1:]
        load_pax(self.stops[bus.last_stop_id], self.time, bus, self.bus_capacity, next_stops,
                 bus.active_trip[0].trip_id)
        return

    def fixed_stop_depart(self, hold=0):
        bus = self.bus
        stops = bus.active_trip[0].stops
        curr_stop_idx = stops.index(bus.last_stop_id)
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME,
                             ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
        dwell_time = (bus.ons + bus.offs > 0) * dwell_time_pax
        if hold:
            dwell_time = max(hold, dwell_time_pax)
        bus.dep_t = self.time + dwell_time
        next_stops = stops[curr_stop_idx + 1:]
        if len(bus.pax) < self.bus_capacity:
            load_pax(self.stops[bus.last_stop_id], bus.dep_t, bus, self.bus_capacity, next_stops,
                     bus.active_trip[0].trip_id, on_dwell=True)
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[bus.last_stop_id] = bus.dep_t
        self.stops[bus.last_stop_id].last_dep_t.append(bus.dep_t)
        runtime = self.get_travel_time(bus.last_stop_id, bus.next_stop_id)
        bus.next_event_time = bus.dep_t + runtime

        last_stop_trip = bus.active_trip[0].stops[-1]
        bus.next_event_type = 2 if bus.next_stop_id == last_stop_trip else 1

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def outbound_ready_to_dispatch(self):
        bus = self.bus
        bus.active_trip.append(bus.pending_trips[0])
        bus.pending_trips.pop(0)
        bus.last_stop_id = bus.active_trip[0].stops[0]
        idx_last_express_stop = STOPS_OUT_FULL_PATT.index(LAST_EXPRESS_STOP)
        if bus.expressed:
            bus.next_stop_id = bus.active_trip[0].stops[idx_last_express_stop + 1]
        else:
            bus.next_stop_id = bus.active_trip[0].stops[1]
        bus.ons = 0
        bus.offs = 0
        bus.denied = 0
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[bus.last_stop_id] = bus.arr_t
        self.stops[bus.last_stop_id].last_arr_t.append(bus.arr_t)
        return

    def outbound_dispatch(self, hold=0):
        bus = self.bus
        stops = bus.active_trip[0].stops
        idx_next_stop = stops.index(bus.next_stop_id)  # ins case there is expressing
        next_stops = stops[idx_next_stop:]

        load_pax(self.stops[bus.last_stop_id], self.time, bus, self.bus_capacity, next_stops,
                 bus.active_trip[0].trip_id)

        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = bus.ons * BOARDING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.ons > 0) * dwell_time
        if hold:
            dwell_time = max(hold, dwell_time)

        bus.dep_t = self.time + dwell_time

        if len(bus.pax) < self.bus_capacity:
            load_pax(self.stops[bus.last_stop_id], bus.dep_t, bus, self.bus_capacity, next_stops,
                     bus.active_trip[0].trip_id,
                     on_dwell=True)
        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[bus.last_stop_id] = bus.dep_t

        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=hold)
        runtime = self.get_travel_time(bus.last_stop_id, bus.next_stop_id)
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 1

        self.stops[bus.last_stop_id].passed_trips.append(bus.active_trip[0].trip_id)
        self.stops[bus.last_stop_id].last_dep_t.append(bus.dep_t)
        return

    def outbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        stops = bus.active_trip[0].stops
        last_stop_trip = stops[-1]
        self.trip_log[trip_idx].stop_arr_times[last_stop_trip] = bus.arr_t
        curr_stop_idx = stops.index(bus.next_stop_id)
        for p in self.bus.pax.copy():
            if p.dest_id == bus.next_stop_id:
                p.alight_time = float(self.time)
                self.completed_pax.append(p)
                self.completed_pax_record.append([p.orig_id, p.dest_id, p.arr_time, p.board_time, p.alight_time,
                                                  p.trip_id, p.denied])
                self.bus.active_trip[0].completed_pax.append(p)
                bus.pax.remove(p)
                bus.offs += 1
            else:
                print(f'what? pax destined for {p.dest_id} (stop {stops.index(p.dest_id)}) '
                      f'and boarded in {p.orig_id} (stop {stops.index(p.orig_id)}) '
                      f'and only {len(stops)} stops served')
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = ACC_DEC_TIME + bus.offs * ALIGHTING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.offs > 0) * dwell_time

        assert len(bus.pax) == 0
        bus.last_stop_id = stops[curr_stop_idx]
        bus.dep_t = float(bus.arr_t) + dwell_time
        self.record_trajectories(offs=bus.offs)
        bus.finished_trips.append(bus.active_trip[0])
        if bus.active_trip[0].trip_id in FOCUS_TRIPS:
            self.focus_trips_finished.append(bus.active_trip[0].trip_id)
        bus.active_trip.pop(0)
        if bus.pending_trips:
            trip = bus.pending_trips[0]
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            next_dep_time = max(bus.dep_t + layover, trip.schedule[0])
            bus.next_event_time = next_dep_time
            bus.next_event_type = 3
            bus.expressed = False
        else:
            bus.deactivate()
        return

    def inbound_dispatch(self):
        bus = self.bus
        bus.active_trip.append(bus.pending_trips[0])
        bus.pending_trips.pop(0)
        trip = bus.active_trip[0]
        trip_id = trip.trip_id
        interval_idx = get_interval(self.time, TRIP_TIME_BIN_MINS) - TRIP_TIME_START_INTERVAL
        rand_percentile = np.random.uniform(0.0, 100.0)
        run_time = np.percentile(RUN_T_DISTR_IN[trip.stops[0] + '-' + trip.stops[-1]][interval_idx], rand_percentile)
        arr_time = self.time + run_time
        start_stop_id = bus.active_trip[0].stops[0]
        bus.dep_t = self.time
        schd_sec = bus.active_trip[0].schedule[0]
        self.in_trip_record.append([trip_id, start_stop_id, bus.dep_t, schd_sec, 1])
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        return

    def inbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_id = bus.active_trip[0].trip_id
        schd_sec = bus.active_trip[0].schedule[-1]
        stop_id = bus.active_trip[0].stops[-1]
        self.in_trip_record.append([trip_id, stop_id, bus.arr_t, schd_sec, len(bus.active_trip[0].stops)])
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            bus.next_event_time = max(self.time + layover, bus.pending_trips[0].schedule[0])
            bus.next_event_type = 0
        else:
            bus.deactivate()
        return

    def next_event(self):
        remaining_buses = [bus for bus in self.buses if bus.active_trip or bus.pending_trips]
        if remaining_buses:
            next_event_times = [bus.next_event_time for bus in remaining_buses]
            min_event_time = min(next_event_times)
            min_event_time_idxs = [i for i, x in enumerate(next_event_times) if x == min_event_time]
            assert min_event_time_idxs
            self.bus = remaining_buses[min_event_time_idxs[0]]
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


class SimulationEnvWithHolding(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(SimulationEnvWithHolding, self).__init__(*args, **kwargs)

    def decide_bus_holding(self):
        # for previous trip
        bus = self.bus
        backward_headway = self.backward_headway()
        last_dep_t = self.stops[bus.last_stop_id].last_dep_t
        if last_dep_t and backward_headway:
            forward_headway = self.time - last_dep_t[-1]
            holding_time = min(LIMIT_HOLDING, max((backward_headway - forward_headway) / 2, 0))
            if self.hold_adj_factor > 0.0 and holding_time > 0:
                holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
        else:
            print(f'what? last arrival time shows nan for controlled trip at time {self.time / 60 / 60}')
            holding_time = 0
        assert holding_time >= 0
        if bus.last_stop_id == STOPS_OUT_FULL_PATT[0]:
            self.outbound_dispatch(hold=holding_time)
        else:
            self.fixed_stop_load()
            self.fixed_stop_depart(hold=holding_time)
        return

    def prep(self):
        done = self.next_event()
        if done or all(elem in self.focus_trips_finished for elem in FOCUS_TRIPS):
            return True

        if self.bus.next_event_type == 0:
            stops = self.bus.pending_trips[0].stops
            trip_id = self.bus.pending_trips[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS and stops == STOPS_OUT_FULL_PATT:
                if stops[0] in CONTROL_STOPS:
                    self.outbound_ready_to_dispatch()
                    self.decide_bus_holding()
                    return False
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
            return False

        if self.bus.next_event_type == 1:
            stops = self.bus.active_trip[0].stops
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS and stops == STOPS_OUT_FULL_PATT:
                if self.bus.next_stop_id in CONTROL_STOPS:
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


class SimulationEnvWithRL(SimulationEnv):
    def __init__(self, estimate_pax=False, weight_ride_t=0.0, weight_cv_hw=0.94, *args, **kwargs):
        super(SimulationEnvWithRL, self).__init__(*args, **kwargs)
        self.trips_sars = {}
        self.bool_terminal_state = False
        self.pool_sars = []
        self.estimate_pax = estimate_pax
        self.weight_ride_t = weight_ride_t
        self.weight_cv_hw = weight_cv_hw

    def reset_simulation(self):
        super().reset_simulation()

        self.bool_terminal_state = False
        self.pool_sars = []

        self.trips_sars = {}
        for trip_id in CONTROL_TRIP_IDS:
            self.trips_sars[trip_id] = []
        return False

    def fixed_stop_load(self, skip=False):
        # REQUIRES RE-WRITING TO ADAPT TO FUNCTION LOAD_PAX
        bus = self.bus
        stops = bus.active_trip[0].stops
        curr_stop_idx = stops.index(bus.last_stop_id)
        bus.denied = 0
        bus.ons = 0
        next_stops = stops[curr_stop_idx + 1:]
        for p in self.stops[bus.last_stop_id].pax.copy():
            if p.arr_time <= self.time and p.dest_id in next_stops:
                if len(bus.pax) + 1 <= self.bus_capacity and not skip:
                    p.trip_id = bus.active_trip[0].trip_id
                    p.board_time = float(self.time)
                    bus.pax.append(p)
                    self.stops[bus.last_stop_id].pax.remove(p)
                    bus.ons += 1
                else:
                    p.denied = 1
                    bus.denied += 1
            else:
                break
        return

    def fixed_stop_depart(self, hold=0, skip=False):
        bus = self.bus
        stops = bus.active_trip[0].stops
        trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        curr_stop_idx = stops.index(bus.last_stop_id)
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        if skip:
            dwell_time = ACC_DEC_TIME + ALIGHTING_TIME * bus.offs + dwell_time_error
            dwell_time = (bus.offs > 0) * dwell_time
        elif hold:
            dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME,
                                 ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
            dwell_time_pax = (bus.ons + bus.offs > 0) * dwell_time_pax
            dwell_time = max(hold, dwell_time_pax)
        else:
            dwell_time = ACC_DEC_TIME + max(bus.ons * BOARDING_TIME, bus.offs * ALIGHTING_TIME) + dwell_time_error
            # herein we zero dwell time if no pax boarded
            dwell_time = (bus.ons + bus.offs > 0) * dwell_time

        bus.dep_t = self.time + dwell_time
        next_stops = stops[curr_stop_idx + 1:]
        if not skip:
            for p in self.stops[bus.last_stop_id].pax.copy():
                if p.arr_time <= bus.dep_t and p.dest_id in next_stops:
                    if len(bus.pax) + 1 <= self.bus_capacity:
                        p.trip_id = bus.active_trip[0].trip_id
                        p.board_time = float(p.arr_time)
                        bus.pax.append(p)
                        self.stops[bus.last_stop_id].pax.remove(p)
                        bus.ons += 1
                    else:
                        p.denied = 1
                        bus.denied += 1
                else:
                    break

        self.trip_log[trip_idx].stop_dep_times[bus.last_stop_id] = bus.dep_t
        runtime = self.get_travel_time(bus.last_stop_id, bus.next_stop_id)
        bus.next_event_time = bus.dep_t + runtime

        last_stop = bus.active_trip[0].stops[-1]
        bus.next_event_type = 2 if bus.next_stop_id == last_stop else 1
        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        bus = self.bus
        stops = bus.active_trip[0].stops
        curr_stop_idx = stops.index(bus.last_stop_id)
        last_arr = self.stops[bus.last_stop_id].last_arr_t[-2]
        second_last_arr = self.stops[bus.last_stop_id].last_arr_t[-3]
        forward_headway = self.time - last_arr
        backward_headway = self.backward_headway()
        trip_id = self.bus.active_trip[0].trip_id
        trip_sars = self.trips_sars[trip_id]
        bus_load = len(self.bus.pax)

        route_progress = curr_stop_idx / len(STOPS_OUT_FULL_PATT)
        if self.estimate_pax:
            interval_idx = get_interval(self.time, ODT_BIN_MINS)
            odt_stop_idx = ODT_STOP_IDS.index(bus.last_stop_id)
            arr_rate = ARR_RATES[interval_idx, odt_stop_idx]
            pax_at_stop = round(forward_headway * (arr_rate / 3600))
        else:
            pax_at_stop = 0
            for p in self.stops[bus.last_stop_id].pax.copy():
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
        # record action ins sars
        trip_id = bus.active_trip[0].trip_id
        self.trips_sars[trip_id][-1][1] = action
        if action:
            hold_time = (action - 1) * BASE_HOLDING_TIME
            if self.hold_adj_factor > 0.0 and hold_time > 0:
                hold_time = np.random.uniform(self.hold_adj_factor * hold_time, hold_time)
                assert hold_time >= 0
            if bus.last_stop_id == STOPS_OUT_FULL_PATT[0]:
                self.outbound_dispatch(hold=hold_time)
            else:
                self.fixed_stop_load()
                self.fixed_stop_depart(hold=hold_time)
        else:
            if bus.last_stop_id == STOPS_OUT_FULL_PATT[0]:
                self.outbound_dispatch()
            else:
                self.fixed_stop_load(skip=True)
                self.fixed_stop_depart(skip=True)
        return

    def delayed_reward(self, s0, s1, neighbor_prev_hold):
        bus = self.bus
        s0_idx = STOPS_OUT_FULL_PATT.index(s0)  # you check from those pax boarding next to the control stop (impacted)
        s1_idx = STOPS_OUT_FULL_PATT.index(s1)  # next control point
        trip_id = bus.active_trip[0].trip_id
        trip_idx = TRIP_IDS_OUT.index(trip_id)

        # FRONT NEIGHBOR'S REWARD
        agent_trip_idx = trip_idx - 1
        agent_trip_id = TRIP_IDS_OUT[agent_trip_idx]
        # agent bus not considered responsible for the wait time impact ins the control point upon arrival to it
        # only responsible for the impact on wait time at the control point for the following trip
        # and the riding time impact ins case of holding which is accounted for later
        # for example if there was skipping at that stop previously the penalty would otherwise be tremendous
        # despite having nothing to do with that decision
        stop_id_before_set = STOPS_OUT_FULL_PATT[:s0_idx]
        stop_id_set = STOPS_OUT_FULL_PATT[s0_idx:s1_idx]
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
            if pax.trip_id == agent_trip_id and pax.orig_id in stop_id_set:
                pax_count += 1
                wait = pax.board_time - pax.arr_time
                ride = min(pax.alight_time - pax.board_time, t1_agent - pax.board_time)
                assert ride > 0
                if pax.orig_id == stop_id_set[0] and wait != 0.0:
                    wait += neighbor_prev_hold
                sum_rew_agent_wait_time += wait
                sum_rew_agent_ride_time += ride
                agent_board_count += 1
            elif pax.trip_id == agent_trip_id and pax.dest_id in stop_id_set[1:]:
                pax_count += 1
                ride = pax.alight_time - t0_agent
                assert ride > 0
                sum_rew_agent_ride_time += ride
        front_active_bus = [bus for bus in self.buses if
                            bus.active_trip and bus.active_trip[0].trip_id == agent_trip_id]
        if front_active_bus:
            for pax in front_active_bus[0].pax:
                if pax.orig_id in stop_id_set:
                    pax_count += 1
                    wait = pax.board_time - pax.arr_time
                    ride = t1_agent - pax.board_time
                    assert ride > 0
                    if pax.orig_id == stop_id_set[0] and wait != 0.0:
                        wait += neighbor_prev_hold
                    sum_rew_agent_wait_time += wait
                    sum_rew_agent_ride_time += ride
                    agent_board_count += 1
                elif pax.orig_id in stop_id_before_set:
                    pax_count += 1
                    ride = t1_agent - t0_agent
                    assert ride > 0
                    sum_rew_agent_ride_time += ride

        # CURRENT TRIP'S REWARD CONTRIBUTION
        # stop_idx_set = [s for s ins range(s0_idx, s1_idx)]
        # +1 is to catch the index of the control point only if it will be used to update
        sum_rew_behind_wait_time = 0
        sum_rew_behind_ride_time = 0
        for pax in bus.pax:
            if pax.orig_id in stop_id_set:
                pax_count += 1
                wait = pax.board_time - pax.arr_time
                sum_rew_behind_wait_time += wait
                ride = t1_behind - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.orig_id in stop_id_before_set:
                pax_count += 1
                ride = t1_behind - t0_behind
                assert ride > 0
                sum_rew_behind_ride_time += ride
        for pax in self.completed_pax:
            if pax.trip_id == trip_id and pax.orig_id in stop_id_set:
                pax_count += 1
                wait = pax.board_time - pax.arr_time
                sum_rew_behind_wait_time += wait
                ride = pax.alight_time - pax.board_time
                assert ride > 0
                sum_rew_behind_ride_time += ride
            elif pax.trip_id == trip_id and pax.dest_id in stop_id_set[1:]:
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

        if trip_id != CONTROL_TRIP_IDS[0] and bus.last_stop_id != CONTROL_STOPS[0]:
            # KEY STEP: WE NEED TO DO TWO THINGS:

            # ONE IS TO FILL IN THE AGENT'S REWARD FOR THE PREVIOUS STEP (IF THERE IS A PREVIOUS STEP)

            # TWO IS TO FILL THE FRONT NEIGHBOR'S REWARD FOR ITS PREVIOUS STEP (IF THERE IS AN ACTIVE FRONT NEIGHBOR)
            # TWO ALSO TRIGGERS SENDING THE SARS TUPLE INTO THE POOL OF COMPLETED EXPERIENCES
            k = CONTROL_STOPS.index(bus.last_stop_id)
            curr_control_stop = CONTROL_STOPS[k]
            prev_control_stop = CONTROL_STOPS[k - 1]
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
                hw_variation = (fw_h1 - planned_fw_h) / planned_fw_h
                if self.hold_adj_factor > 0.0:
                    prev_hold = prev_hold - (prev_hold - self.hold_adj_factor * prev_hold) / 2
                    reward = - weight_cv_hw * hw_variation * hw_variation - (1 - weight_cv_hw) * (
                            prev_hold / LIMIT_HOLDING)
                else:
                    reward = - weight_cv_hw * hw_variation * hw_variation - (1 - weight_cv_hw) * (
                            prev_hold / LIMIT_HOLDING)
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
            stops = self.bus.pending_trips[0].stops
            trip_id = self.bus.pending_trips[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS and stops == STOPS_OUT_FULL_PATT:
                if stops[0] in CONTROL_STOPS[:-1]:
                    self.bool_terminal_state = False
                    self.outbound_ready_to_dispatch()
                    self._add_observations()
                    return False
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
            return self.prep()

        if self.bus.next_event_type == 1:
            stops = self.bus.active_trip[0].stops
            stop = self.bus.next_stop_id
            trip_id = self.bus.active_trip[0].trip_id
            if trip_id not in NO_CONTROL_TRIP_IDS and stops == STOPS_OUT_FULL_PATT:
                if stop in CONTROL_STOPS:
                    self.bool_terminal_state = False
                    self.fixed_stop_unload()
                    self._add_observations()
                    return False
                elif stop == TERMINAL_STATE_STOP:
                    self.bool_terminal_state = True
                    self.fixed_stop_unload()
                    self._add_observations()
                    self.fixed_stop_load()
                    self.fixed_stop_depart()
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


class SimulationEnvWithCancellations(SimulationEnv):
    def __init__(self, weight_hold_t=0.0, control_strategy=None, cancelled_blocks=None,
                 *args, **kwargs):
        super(SimulationEnvWithCancellations, self).__init__(*args, **kwargs)
        self.weight_hold_t = weight_hold_t
        self.control_strategy = control_strategy
        self.cancelled_blocks = cancelled_blocks
        self.obs = []
        self.prev_obs = []
        self.prev_reward = None
        self.prev_hold_t = None
        self.prev_decision_t = None
        self.moved_up_trips = None

    def reset_simulation(self):
        self.time = START_TIME_SEC
        self.focus_trips_finished = []
        self.bus = Bus(0, [])

        # for records
        self.out_trip_record = []
        self.in_trip_record = []

        # initialize buses (we treat each block as a separate bus)
        self.buses = []
        for block_trip_set in BLOCK_TRIPS_INFO:
            block_id = block_trip_set[0]
            trip_set = block_trip_set[1]
            self.buses.append(Bus(block_id, trip_set))
        # initialize passenger demand
        self.completed_pax = []
        self.completed_pax_record = []
        self.initialize_pax_demand()
        # initialize bus trips (the first trip for each bus)
        if self.control_strategy == 'EDS':
            self.moved_up_trips = []
        for bus in self.buses:
            if self.cancelled_blocks:
                if bus.bus_id in self.cancelled_blocks:
                    bus.cancelled = True
            if bus.cancelled:
                bus.cancelled_trips = deepcopy(bus.pending_trips)
                if self.control_strategy == 'EDS':
                    # early dispatchiing strategy - depart early the follower of missing trip
                    # pre-planned therefore we add these trips to a list
                    cancelled_trip_ids = [t.trip_id for t in bus.cancelled_trips]
                    for t in cancelled_trip_ids:
                        if t in TRIP_IDS_OUT[:-1]:
                            self.moved_up_trips.append(TRIP_IDS_OUT[TRIP_IDS_OUT.index(t) + 1])
                bus.pending_trips = []
                bus.deactivate()
            else:
                for trip in bus.pending_trips:
                    self.trip_log.append(TripLog(trip.trip_id, trip.stops))
                first_trip = bus.pending_trips[0]
                interval_idx = get_interval(first_trip.schedule[0], DELAY_BIN_MINS) - DELAY_START_INTERVAL
                rand_percentile = np.random.uniform(0.0, 100.0)
                if first_trip.direction == 0:
                    delay_dist = DELAY_DISTR_OUT[interval_idx]
                else:
                    delay_dist = DELAY_DISTR_IN[first_trip.stops[0]][interval_idx]  # DICTIONARY TYPE
                delay = np.percentile(delay_dist, rand_percentile)
                if first_trip.direction:
                    bus.next_event_time = first_trip.schedule[0] + max(delay, 0)  # delay can be negative
                    bus.next_event_type = 3
                else:
                    bus.next_event_time = max(first_trip.schedule[0] - EARLY_DEP_LIMIT_SEC,
                                              first_trip.schedule[0] + delay)
                    # delay can be negative
                    bus.next_event_type = 5

        self.obs = []
        self.prev_obs = []
        self.prev_reward = None
        self.prev_hold_t = None
        self.prev_decision_t = None
        return False

    def get_express_run_time(self, stop0, stop1):
        bus = self.bus
        stops = bus.active_trip[0].stops
        idx0 = stops.index(stop0)
        idx1 = stops.index(stop1)
        run_t = 0
        for i in range(idx0, idx1):
            run_t += self.get_travel_time(stops[i], stops[i + 1])
        return run_t

    def inbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        trip_id = bus.active_trip[0].trip_id
        stop_id = bus.active_trip[0].stops[-1]
        schd_sec = bus.active_trip[0].schedule[-1]
        self.in_trip_record.append([trip_id, stop_id, bus.arr_t, schd_sec, len(bus.active_trip[0].stops)])
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            # NEXT EVENT IS THE EARLIEST BETWEEN THE READY TIME AND THE EARLY DEPARTURE LIMIT
            ready_time = self.time + layover
            bus.next_event_time = max(ready_time, bus.pending_trips[0].schedule[0] - EARLY_DEP_LIMIT_SEC)
            bus.next_event_type = 5
        else:
            bus.deactivate()
        return
    

    def actual_future_headways(self):
        # terminal
        # search for active buses inbound and compute the time they will pick up pax at terminal (arrival time)
        future_dep_t = []
        # this will collect the expected arrival time of all trips departing
        for bus in self.buses:
            if bus.active_trip and bus.active_trip[0].direction == 1 \
                    and bus.pending_trips and bus.bus_id != self.bus.bus_id:
                interv_idx = get_interval(bus.dep_t, TRIP_TIME_BIN_MINS) - TRIP_TIME_START_INTERVAL
                stops = bus.active_trip[0].stops
                estimated_in_run_t = np.mean(RUN_T_DISTR_IN[stops[0] + '-' + stops[-1]][interv_idx])
                ready_time = bus.dep_t + estimated_in_run_t + MIN_LAYOVER_T
                next_sched_t = bus.pending_trips[0].schedule[0]
                t = max(ready_time, next_sched_t, self.time+5)
                future_dep_t.append(t)
            if not bus.active_trip and bus.pending_trips and bus.bus_id != self.bus.bus_id:
                if bus.pending_trips[0].direction == 0:
                    if bus.instructed_hold_time:
                        # the bus could 'hold' to a time earlier than its scheduled departure
                        future_dep_t.append(bus.next_event_time)
                        assert bus.next_event_time >= self.time
                    else:
                        if not bus.finished_trips:
                            # pull-out trip
                            future_dep_t.append(max(self.time+5,bus.pending_trips[0].schedule[0]))
                        else:
                            # bus layover at terminal without holding instruction
                            t = max(bus.next_event_time, bus.pending_trips[0].schedule[0])

                            future_dep_t.append(t)
                elif len(bus.pending_trips) > 1:
                    assert bus.pending_trips[1].direction == 0
                    future_dep_t.append(bus.pending_trips[1].schedule[0])
        if len(future_dep_t) > 0:
            future_dep = sorted(future_dep_t)
            future_dep.insert(0, self.time)
            future_dep_hw = [future_dep[i] - future_dep[i - 1] for i in range(1, len(future_dep))]
            return future_dep_hw
        else:
            return []

    def report_observations(self):
        bus = self.bus
        stops = bus.pending_trips[0].stops
        sched_dep = bus.pending_trips[0].schedule[0]
        # this will be the reference point - future and past hw extracted based on
        past_sched_dep = np.sort(sched_deps_arr[sched_deps_arr <= sched_dep])[:].tolist()
        past_sched_hw = [past_sched_dep[i] - past_sched_dep[i - 1] for i in range(1, len(past_sched_dep))]

        past_actual_dep = deepcopy(self.stops[stops[0]].last_dep_t)
        past_actual_dep.append(self.time)
        past_actual_hw = [past_actual_dep[i] - past_actual_dep[i - 1] for i in range(1, len(past_actual_dep))]
        # print([time_string(past_actual_dep[i]) for i ins range(len(past_actual_dep))])

        future_sched_dep = np.sort(sched_deps_arr[sched_deps_arr >= sched_dep])[:FUTURE_HW_HORIZON + 1].tolist()
        future_sched_hw = [future_sched_dep[i] - future_sched_dep[i - 1] for i in range(1, len(future_sched_dep))]

        future_actual_hw = self.actual_future_headways()
        sched_dev = self.time - sched_dep

        if len(past_actual_hw) >= PAST_HW_HORIZON and len(future_actual_hw) >= FUTURE_HW_HORIZON:
            self.obs = past_actual_hw[-PAST_HW_HORIZON:] + future_actual_hw[:FUTURE_HW_HORIZON] + past_sched_hw[
                                                                                                  -PAST_HW_HORIZON:] + future_sched_hw + [
                           sched_dev]
            assert len(self.obs) == PAST_HW_HORIZON * 2 + FUTURE_HW_HORIZON * 2 + 1
            if self.prev_obs:
                # BASELINE REWARD
                prev_bw_h = self.prev_obs[PAST_HW_HORIZON]
                prev_fw_h = self.prev_obs[PAST_HW_HORIZON - 1]
                prev_sched_dev = self.prev_obs[-1]
                prev_hold_t_max = max(IMPOSED_DELAY_LIMIT - prev_sched_dev, 0)
                hold_t = max(-prev_sched_dev, 0)

                predicted_fw_h = prev_fw_h + hold_t
                predicted_bw_h = max(self.time, sched_dep) - (self.prev_decision_t + hold_t)

                rew_baseline = -np.power((predicted_fw_h - predicted_bw_h) / 1000, 2)

                # RL REWARD
                sched_dev = deepcopy(self.obs[-1])
                hold_time = deepcopy(self.prev_hold_t)
                resulting_fw_h = self.obs[PAST_HW_HORIZON - 2]
                resulting_bw_h = self.obs[PAST_HW_HORIZON - 1] - min(sched_dev,
                                                                     0)  # to add the component of early departure
                rew_rl = -np.power((resulting_fw_h - resulting_bw_h) / 1000, 2)

                if abs(hold_time - hold_t) > (HOLD_INTERVALS / 2):
                    self.prev_reward = rew_rl - rew_baseline
                else:
                    self.prev_reward = 0.0

        elif len(past_actual_hw) > 0 and self.control_strategy != 'NC' and len(future_actual_hw) > 0:
            hold_t = even_hw_decision(future_actual_hw[0], past_actual_hw[-1], max(IMPOSED_DELAY_LIMIT - sched_dev, 0))
            bus.next_event_time = self.time + hold_t
            bus.next_event_type = 0
            bus.instructed_hold_time = hold_t
        else:
            bus.next_event_time = max(self.time, sched_dep)
            bus.next_event_type = 0

    def dispatch_decision(self, hold_time=None):
        bus = self.bus
        sched_dep = bus.pending_trips[0].schedule[0]
        if self.control_strategy != 'NC':
            obs = self.obs
            sched_dev = obs[-1]
            hold_time_max = max(IMPOSED_DELAY_LIMIT - sched_dev, 0)
            fw_h = obs[PAST_HW_HORIZON - 1]
            assert self.control_strategy in ['EDS', 'HDS', 'HDS+MRH', 'HDS+TX', 'RL']
            if self.control_strategy == 'EDS':
                if bus.pending_trips[0].trip_id in self.moved_up_trips:
                    hold_time = 0.0
                    sched_fw_h = obs[PAST_HW_HORIZON+FUTURE_HW_HORIZON+PAST_HW_HORIZON]
                    bus.next_event_time = max(self.time, sched_dep - sched_fw_h/2)
                    # print(f'----')
                    # print(f'{time_string(self.time)}')
                    # print(f'current headway {time_string(sched_fw_h)}')
                    # print(f'scheduled departure {time_string(sched_dep)}')
                    # print(f'departure {time_string(bus.next_event_time)}')
                else:
                    hold_time = max(sched_dep-self.time, 0)
                    bus.next_event_time = max(self.time, sched_dep)
            if self.control_strategy in ['HDS', 'HDS+MRH']:
                bw_h = obs[PAST_HW_HORIZON]
                hold_time = even_hw_decision(bw_h, fw_h, hold_time_max)
                # print('HOLDING DECISION')
                # print(f'Hold time of {round(hold_time)} <= {round(hold_time_max)}')
                bus.next_event_time = self.time + hold_time
            if self.control_strategy == 'HDS+TX':
                sched_fw_h = obs[PAST_HW_HORIZON + FUTURE_HW_HORIZON + PAST_HW_HORIZON - 1]
                bw_h = obs[PAST_HW_HORIZON]
                sched_bw_h = obs[PAST_HW_HORIZON * 2 + FUTURE_HW_HORIZON]
                fw_h_condition = fw_h >= sched_fw_h * FW_H_LIMIT_EXPRESS
                fw_bw_h_condition = (fw_h >= bw_h * BF_H_LIMIT_EXPRESS)
                bw_h_condition = bw_h <= sched_bw_h * BW_H_LIMIT_EXPRESS
                assert bus.expressed is False
                if (fw_h_condition or bw_h_condition) and fw_bw_h_condition:
                    hold_time = even_hw_decision(bw_h, fw_h, hold_time_max)
                    assert hold_time == 0
                    bus.expressed = True
                else:
                    hold_time = even_hw_decision(bw_h, fw_h, hold_time_max)
                    bus.next_event_time = self.time + hold_time
                bus.next_event_time = self.time + hold_time
            if self.control_strategy == 'RL':
                bus.next_event_time = self.time + hold_time
            bus.instructed_hold_time = hold_time
            self.prev_obs = deepcopy(self.obs)
            self.prev_decision_t = deepcopy(self.time)
            self.prev_hold_t = deepcopy(hold_time)
        else:
            bus.next_event_time = max(self.time, sched_dep)
        self.obs = []
        bus.next_event_type = 0
        return

    def mid_rt_hold_decision(self):
        # for previous trip
        bus = self.bus
        backward_headway = self.backward_headway()
        last_dep_t = self.stops[bus.last_stop_id].last_dep_t
        if last_dep_t and backward_headway:
            forward_headway = self.time - last_dep_t[-1]
            holding_time = even_hw_decision(backward_headway, forward_headway, MRH_HOLD_LIMIT)
            # print('HOLDING DECISION')
            # print(f'at time {str(timedelta(seconds=round(self.time)))} at stop {bus.last_stop_id}')
            # print([str(timedelta(seconds=round(forward_headway))), str(timedelta(seconds=round(backward_headway)))])
            # print(f'Hold time of {round(holding_time)} <= {round(MRH_HOLD_LIMIT)}')
        else:
            # print(f'what? last arrival time shows nan for controlled trip at time {self.time / 60 / 60}')
            holding_time = 0
        assert holding_time >= 0
        return holding_time

    def outbound_dispatch(self, hold=0):
        bus = self.bus
        stops = bus.active_trip[0].stops
        idx_next_stop = stops.index(bus.next_stop_id)
        next_stops = stops[idx_next_stop:]

        load_pax(self.stops[bus.last_stop_id], self.time, bus, self.bus_capacity, next_stops,
                 bus.active_trip[0].trip_id)

        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time = bus.ons * BOARDING_TIME + dwell_time_error
        # herein we zero dwell time if no pax boarded
        dwell_time = (bus.ons > 0) * dwell_time
        if bus.instructed_hold_time:
            bus.dep_t = self.time + max(dwell_time - bus.instructed_hold_time, 0)
            bus.instructed_hold_time = None
        else:
            bus.dep_t = self.time + dwell_time
        if len(bus.pax) < self.bus_capacity:
            load_pax(self.stops[bus.last_stop_id], bus.dep_t, bus, self.bus_capacity, next_stops,
                     bus.active_trip[0].trip_id,
                     on_dwell=True)

        curr_trip_idx = TRIP_IDS_OUT.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[bus.last_stop_id] = bus.dep_t
        if bus.instructed_hold_time:
            self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=bus.instructed_hold_time)
        else:
            self.record_trajectories(pickups=bus.ons, denied_board=bus.denied)
        if bus.expressed:
            runtime = self.get_express_run_time(bus.last_stop_id, bus.next_stop_id)
        else:
            runtime = self.get_travel_time(bus.last_stop_id, bus.next_stop_id)
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 1

        self.stops[bus.last_stop_id].passed_trips.append(bus.active_trip[0].trip_id)
        self.stops[bus.last_stop_id].last_dep_t.append(bus.dep_t)
        return

    def prep(self) -> object:
        done = self.next_event()
        if done or self.time >= END_TIME_SEC - 60 * 30:
            return True

        if self.bus.next_event_type == 0:
            self.outbound_ready_to_dispatch()
            self.outbound_dispatch()
            return self.prep()

        if self.bus.next_event_type == 1:
            # conditions: strategy has MRH, bus is not expressed, bus has something behind (ins function)
            # and is a controlled stop
            self.fixed_stop_unload()
            if self.control_strategy != 'NC' and 'MRH' in self.control_strategy:
                cond2 = not self.bus.expressed
                cond3 = self.bus.active_trip[0].stops == STOPS_OUT_FULL_PATT
                cond4 = self.bus.last_stop_id in MRH_STOPS
                cond = cond2 and cond3 and cond4
                if cond:
                    ht = self.mid_rt_hold_decision()
                    self.fixed_stop_load()
                    self.fixed_stop_depart(hold=ht)
                else:
                    self.fixed_stop_load()
                    self.fixed_stop_depart()
            else:
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

        if self.bus.next_event_type == 5:  # outbound terminal dispatch decision point
            self.report_observations()
            if self.obs:
                return False
            else:
                return self.prep()
