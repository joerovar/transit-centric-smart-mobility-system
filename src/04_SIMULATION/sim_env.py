from input import *
import numpy as np
from scipy.stats import lognorm
import random
from copy import deepcopy
from agents_sim import Passenger, Stop, Bus, Log, TripLog
from datetime import timedelta


def run_base_detailed(replications=4, save_results=False, time_dep_tt=True, time_dep_dem=True):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    cv_set = []
    for _ in range(replications):
        env = DetailedSimulationEnv(time_dependent_travel_time=time_dep_tt, time_dependent_demand=time_dep_dem)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        cv_per_stop = []
        for s in env.stops:
            arr_times = np.array(s.last_arr_t)
            arr_times = arr_times[(arr_times>FOCUS_START_TIME_SEC) & (arr_times<FOCUS_END_TIME_SEC+3600)]
            arr_times = arr_times.tolist()
            hw = [arr_times[i] - arr_times[i-1] for i in range(1, len(arr_times))]
            cv = np.std(hw) / np.mean(hw)
            cv_per_stop.append(cv)
        cv_set.append(cv_per_stop)
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    plt.plot(np.array(cv_set).mean(axis=0))
    plt.show()
    plt.close()
    if save_results:
        path_trajectories = 'out/NC/'+tstamp+'-trajectory_set' + ext_var
        path_completed_pax = 'out/NC/'+tstamp+'-pax_set' + ext_var
        save(path_trajectories, trajectories_set)
        save(path_completed_pax, pax_set)
    return


def run_base_control_detailed(replications=2, control_strength=0.7,
                              save_results=False, time_dep_tt=True, time_dep_dem=True, hold_adj_factor=0.0,
                              tt_factor=1.0):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    trajectories_set = []
    pax_set = []
    for _ in range(replications):
        env = DetailedSimulationEnvWithControl(time_dependent_travel_time=time_dep_tt,
                                               time_dependent_demand=time_dep_dem, hold_adj_factor=hold_adj_factor,
                                               tt_factor=tt_factor)
        done = env.reset_simulation()
        while not done:
            done = env.prep()
        if save:
            env.process_results()
            trajectories_set.append(env.trajectories)
            pax_set.append(env.completed_pax)
    if save_results:
        path_trajectories = 'out/EH/' + tstamp + '-trajectory_set' + ext_var
        path_completed_pax = 'out/EH/' + tstamp + '-pax_set' + ext_var
        params = {'param': ['control_strength'],
                  'value': [control_strength]}
        df_params = pd.DataFrame(params)
        df_params.to_csv('out/EH/' + tstamp + '-params_used' + ext_var, index=False)
        save(path_trajectories, trajectories_set)
        save(path_completed_pax, pax_set)
    return


def run_sample_rl(episodes=1, simple_reward=False, weight_ride_t=0.0):
    tstamp = datetime.now().strftime('%m%d-%H%M%S')
    for _ in range(episodes):
        env = DetailedSimulationEnvWithDeepRL(estimate_pax=True, weight_ride_t=weight_ride_t)
        done = env.reset_simulation()
        done = env.prep()
        while not done:
            trip_id = env.bus.active_trip[0].trip_id
            all_sars = env.trips_sars[trip_id]
            if not env.bool_terminal_state:
                observation = np.array(all_sars[-1][0], dtype=np.float32)
                route_progress = observation[IDX_RT_PROGRESS]
                pax_at_stop = observation[IDX_PAX_AT_STOP]
                curr_stop = [s for s in env.stops if s.stop_id == env.bus.last_stop_id]
                previous_denied = False
                for p in curr_stop[0].pax.copy():
                    if p.arr_time <= env.time:
                        if p.denied:
                            previous_denied = True
                            break
                    else:
                        break
                if route_progress == 0.0 or pax_at_stop == 0 or previous_denied:
                    action = random.randint(1, 4)
                else:
                    action = random.randint(0, 4)
                env.take_action(action)
            env.update_rewards(simple_reward=simple_reward)
            done = env.prep()
    return


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
    def __init__(self, no_overtake_policy=True, time_dependent_travel_time=True, time_dependent_demand=True,
                 tt_factor=1.0, hold_adj_factor=0.0):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        # self.no_overtake_policy = no_overtake_policy
        self.time_dependent_travel_time = time_dependent_travel_time
        self.time_dependent_demand = time_dependent_demand
        self.time = 0.0
        # RECORDINGS
        self.trajectories = {}
        self.tt_factor = tt_factor
        self.hold_adj_factor = hold_adj_factor


class DetailedSimulationEnv(SimulationEnv):
    def __init__(self, *args, **kwargs):
        super(DetailedSimulationEnv, self).__init__(*args, **kwargs)
        self.stops = []
        self.completed_pax = []
        self.buses = []
        self.bus = Bus(0, [])
        self.log = Log(TRIP_IDS_IN + TRIP_IDS_OUT)
        self.trip_log = []

    def backward_headway(self):
        # terminal
        # search for active buses inbound and compute the time they will pick up pax at terminal (arrival time)
        expected_arr_t = []
        # this will collect the expected arrival time of all trips departing
        for bus in self.buses:
            # HERE WE TAKE TRIPS THAT ARE MID-ROUTE IN THE INBOUND DIRECTION
            if bus.active_trip and bus.active_trip[0].route_type in [1, 2] and bus.pending_trips:
                assert bus.next_event_type == 4 or bus.next_event_type == 3
                prev_dep_t = max(bus.dep_t, bus.active_trip[0].sched_time)
                interv_idx = get_interval(prev_dep_t, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
                next_sched_t = bus.pending_trips[0].sched_time
                if bus.active_trip[0].route_type == 1:
                    estimated_in_run_t = np.mean(TRIP_T1_DIST_IN[interv_idx])
                else:
                    estimated_in_run_t = np.mean(TRIP_T2_DIST_IN[interv_idx])
                ready_time = prev_dep_t + estimated_in_run_t + MIN_LAYOVER_T
                if self.bus.last_stop_id == STOPS_OUTBOUND[0]:
                    expected_arr_t.append(max(ready_time, next_sched_t))
                else:
                    terminal_dep_t = max(ready_time, next_sched_t)
                    # print(f'the terminal dep time is between ready time {ready_time/60/60} and scheduled time {next_sched_t/60/60}')
                    arr_t = estimate_arrival_time(terminal_dep_t, STOPS_OUTBOUND[0], self.bus.last_stop_id, self.time_dependent_travel_time, self.time)
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
                    arr_t = estimate_arrival_time(terminal_dep_t, STOPS_OUTBOUND[0], self.bus.last_stop_id, self.time_dependent_travel_time, self.time)
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
                            print(f'behind bus departed {round((self.time - bus.dep_t)/60, 1)} mins ago but is expected to run for {round((arr_t - bus.dep_t)/60, 1)}')
                        if arr_t - self.time >= 0:
                            expected_arr_t.append(arr_t)

        if expected_arr_t:
            return min(expected_arr_t) - self.time
        else:
            # this might happen if the episode is still running and we control the last dispatched trip
            print(f'no future arrival found at hour {self.time/60/60}')
            return SCHED_DEP_IN[-1] - SCHED_DEP_IN[-2]

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
        self.bus = Bus(0, [])
        # for records
        self.trajectories = {}
        for trip_id in TRIP_IDS_IN:
            self.trajectories[trip_id] = []
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
            interval_delay = get_interval(trip.sched_time, DELAY_INTERVAL_LENGTH_MINS) - DELAY_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            if trip.route_type == 0:
                delay = np.percentile(DEP_DELAY_DIST_OUT[interval_delay], rand_percentile)
            elif trip.route_type == 1:
                delay = np.percentile(DEP_DELAY1_DIST_IN[interval_delay], rand_percentile)
            else:
                assert trip.route_type == 2
                delay = np.percentile(DEP_DELAY2_DIST_IN[interval_delay], rand_percentile)
            # random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.sched_time + max(delay, 0)
            bus.dep_t = deepcopy(bus.next_event_time)
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
                    od_rate = SCALED_ODT_RATES[interval_idx, odt_orig_idx, odt_dest_idx]
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
        # print([counter, round(counter_od_interv_1, 1), round(counter_od_interv_2, 1)])
        return

    def fixed_stop_unload(self):
        # SWITCH POSITION AND ARRIVAL TIME
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.next_stop_id)
        bus.last_stop_id = STOPS_OUTBOUND[curr_stop_idx]
        bus.next_stop_id = STOPS_OUTBOUND[curr_stop_idx + 1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.arr_t
        self.stops[curr_stop_idx].last_arr_t.append(bus.arr_t)
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
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
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
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        dwell_time_error = max(random.uniform(-DWELL_TIME_ERROR, DWELL_TIME_ERROR), 0)
        dwell_time_pax = max(ACC_DEC_TIME + bus.ons * BOARDING_TIME,
                             ACC_DEC_TIME + bus.offs * ALIGHTING_TIME) + dwell_time_error
        dwell_time = (bus.ons + bus.offs > 0) * dwell_time_pax
        if hold:
            dwell_time = max(hold, dwell_time_pax)
        bus.dep_t = self.time + dwell_time

        # if self.no_overtake_policy and curr_trip_idx:
        #     prev_dep_t = self.trip_log[curr_trip_idx-1].stop_dep_times[bus.last_stop_id]
        #     bus.dep_t = max(prev_dep_t + NO_OVERTAKE_BUFFER, bus.dep_t)

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

        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.dep_t
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        bus.next_event_type = 2 if bus.next_stop_id == STOPS_OUTBOUND[-1] else 1
        # if self.no_overtake_policy:
        #     bus.next_event_time = deepcopy(self.no_overtake())

        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold)
        return

    def outbound_ready_to_dispatch(self):
        bus = self.bus
        bus.last_stop_id = STOPS_OUTBOUND[0]
        bus.next_stop_id = STOPS_OUTBOUND[1]
        bus.arr_t = self.time
        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_arr_times[STOPS_OUTBOUND[0]] = bus.arr_t
        self.stops[0].last_arr_t.append(bus.arr_t)
        return

    def outbound_dispatch(self, hold=0):
        bus = self.bus
        # trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        bus.denied = 0
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

        # if self.no_overtake_policy and trip_idx:
        #     prev_dep_t = self.trip_log[trip_idx-1].stop_dep_times[STOPS_OUTBOUND[0]]
        #     if prev_dep_t is None:
        #         print('---report')
        #         print(f'trips in question {TRIP_IDS_IN[trip_idx - 1:trip_idx + 1]}')
        #         print(f'time {round(self.time)}')
        #         print(f'stop {0}')
        #     bus.dep_t = max(bus.dep_t, prev_dep_t + NO_OVERTAKE_BUFFER)

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

        curr_trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        self.trip_log[curr_trip_idx].stop_dep_times[STOPS_OUTBOUND[0]] = bus.dep_t

        self.record_trajectories(pickups=bus.ons, denied_board=bus.denied, hold=hold)
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime
        # if self.no_overtake_policy:
        #     bus.next_event_time = deepcopy(self.no_overtake())
        bus.next_event_type = 1

        self.stops[0].passed_trips.append(bus.active_trip[0].trip_id)
        self.log.recorded_departures[self.bus.active_trip[0].trip_id] = bus.dep_t
        return

    def outbound_arrival(self):
        bus = self.bus
        bus.arr_t = self.time
        self.log.recorded_arrivals[self.bus.active_trip[0].trip_id] = bus.arr_t
        trip_idx = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        self.trip_log[trip_idx].stop_arr_times[STOPS_OUTBOUND[-1]] = bus.arr_t
        curr_stop_idx = STOPS_OUTBOUND.index(bus.next_stop_id)
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
        bus.last_stop_id = STOPS_OUTBOUND[curr_stop_idx]
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
                # this never happens
                assert route_type == 2
                interval = get_interval(bus.dep_t, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
                mean, std = DEADHEAD_TIME_PARAMS[interval]
                deadhead_time = norm.rvs(loc=mean, scale=std)
                next_dep_time = max(bus.dep_t + deadhead_time, trip.sched_time)
            bus.next_event_time = next_dep_time
            bus.next_event_type = 3
        return

    def inbound_dispatch(self):
        bus = self.bus
        trip = bus.active_trip[0]
        route_type = trip.route_type
        # types {1: long, 2: short}
        if route_type == 1:
            interval_idx = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            # trip_time_params = TRIP_TIMES1_PARAMS[interval]
            # run_time = lognorm.rvs(*trip_time_params)
            rand_percentile = np.random.uniform(0.0, 100.0)
            run_time = np.percentile(TRIP_T1_DIST_IN[interval_idx], rand_percentile)
            arr_time = self.time + run_time
        else:
            assert route_type == 2
            interval_idx = get_interval(self.time, TRIP_TIME_INTERVAL_LENGTH_MINS) - TRIP_TIME_START_INTERVAL
            # trip_time_params = TRIP_TIMES2_PARAMS[interval]
            # run_time = lognorm.rvs(*trip_time_params)
            rand_percentile = np.random.uniform(0.0, 100.0)
            run_time = np.percentile(TRIP_T2_DIST_IN[interval_idx], rand_percentile)
            arr_time = self.time + run_time
        bus.dep_t = self.time
        bus.next_event_time = arr_time
        bus.next_event_type = 4
        self.log.recorded_departures[trip.trip_id] = bus.dep_t
        return

    def no_overtake(self):
        next_instance_t = deepcopy(self.bus.next_event_time)
        trip_idx = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        next_stop_id = self.bus.next_stop_id
        if trip_idx:
            leader_arr_t_next_stop = self.trip_log[trip_idx-1].stop_arr_times[next_stop_id]
            if leader_arr_t_next_stop is None:
                prev_trip_id = TRIP_IDS_IN[trip_idx - 1]
                active_buses = [bus for bus in self.buses if bus.active_trip]
                leading_bus = [bus for bus in active_buses if bus.active_trip[0].trip_id == prev_trip_id]
                assert next_stop_id == leading_bus[0].next_stop_id
                next_instance_t = max(next_instance_t, leading_bus[0].next_event_time + NO_OVERTAKE_BUFFER)
            else:
                next_instance_t = max(next_instance_t, leader_arr_t_next_stop + NO_OVERTAKE_BUFFER)
        return next_instance_t

    def inbound_arrival(self):
        bus = self.bus
        bus.finished_trips.append(bus.active_trip[0])
        bus.active_trip.pop(0)
        if bus.pending_trips:
            # self.smart_dispatch()
            bus.active_trip.append(bus.pending_trips[0])
            bus.pending_trips.pop(0)
            bus.last_stop_id = STOPS_OUTBOUND[0]
            layover = MIN_LAYOVER_T + max(np.random.uniform(-ERR_LAYOVER_TIME, ERR_LAYOVER_TIME), 0)
            bus.next_event_time = max(self.time + layover, bus.active_trip[0].sched_time)
            bus.next_event_type = 0
        bus.arr_t = self.time
        self.log.recorded_arrivals[self.bus.finished_trips[-1].trip_id] = bus.arr_t
        return

    # def smart_dispatch(self):
    #     # check if previous trip has departed by the ready to dispatch time of the current trip else SWITCH
    #     bus = self.bus
    #     trip = bus.pending_trips[0]
    #     trip_idx = TRIP_IDS_IN.index(trip.trip_id)
    #     if trip_idx:
    #         prev_trip_idx_passed = TRIP_IDS_IN.index(self.stops[0].passed_trips[-1])
    #         if prev_trip_idx_passed < trip_idx - 1:
    #             for prev_trip_idx in range(prev_trip_idx_passed+1, trip_idx):
    #                 prev_trip_id = TRIP_IDS_IN[prev_trip_idx]
    #                 lead_bus_id = next(key for key, value_lst in BLOCK_DICT.items() if prev_trip_id in value_lst)
    #                 lead_bus = next(bu for bu in self.buses if bu.bus_id == lead_bus_id)
    #                 lead_bus_pending_trip_ids = [t.trip_id for t in lead_bus.pending_trips]
    #                 if prev_trip_id in lead_bus_pending_trip_ids:
    #                     switch_idx = lead_bus_pending_trip_ids.index(prev_trip_id)
    #                     trips_bus = bus.pending_trips
    #                     bus_id = bus.bus_id
    #                     trips_lead_bus = lead_bus.pending_trips
    #
    #                     bus.bus_id = lead_bus_id
    #                     bus.pending_trips = trips_lead_bus[switch_idx:]
    #                     lead_bus.bus_id = bus_id
    #                     lead_bus.pending_trips = trips_lead_bus[:switch_idx] + trips_bus
    #                     break
    #     return

    # def smart_dispatch_initialized(self):
    #     bus = self.bus
    #     trip = self.bus.active_trip[0]
    #     trip_idx = TRIP_IDS_IN.index(trip.trip_id)
    #     if trip_idx:
    #         prev_trip_idx_passed = TRIP_IDS_IN.index(self.stops[0].passed_trips[-1])
    #         if prev_trip_idx_passed < trip_idx - 1:
    #             for prev_trip_idx in range(prev_trip_idx_passed, trip_idx):
    #                 prev_trip_id = TRIP_IDS_IN[prev_trip_idx]
    #                 lead_bus_id = next(key for key, value_lst in BLOCK_DICT.items() if prev_trip_id in value_lst)
    #                 lead_bus = next(bu for bu in self.buses if bu.bus_id == lead_bus_id)
    #                 lead_bus_pending_trip_ids = [t.trip_id for t in lead_bus.pending_trips]
    #                 if prev_trip_id in lead_bus_pending_trip_ids:
    #                     switch_idx = lead_bus_pending_trip_ids.index(prev_trip_id)
    #                     trips_bus = bus.pending_trips
    #                     bus_id = bus.bus_id
    #                     trips_lead_bus = lead_bus.pending_trips
    #                     bus.bus_id = lead_bus_id
    #                     bus.active_trip[0] = trips_lead_bus[switch_idx]
    #                     if switch_idx > len(trips_lead_bus) - 1:
    #                         bus.pending_trips = trips_lead_bus[switch_idx+1:]
    #                     else:
    #                         bus.pending_trips = []
    #                     lead_bus.bus_id = bus_id
    #                     lead_bus.pending_trips = trips_lead_bus[:switch_idx] + trips_bus
    #                     break
    #     return

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
            # if not self.bus.finished_trips:
            #     self.smart_dispatch_initialized()
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
        stop = self.stops[STOPS_OUTBOUND.index(self.bus.last_stop_id)]
        idx_trip_id = TRIP_IDS_IN.index(self.bus.active_trip[0].trip_id)
        if stop.stop_id == STOPS_OUTBOUND[0]:
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
                    assert stop0 == STOPS_OUTBOUND[0]
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
                stop0 = STOPS_OUTBOUND[0]
            stop1 = self.bus.last_stop_id
            behind_trip_arrival_time = estimate_arrival_time(dep_t, stop0, stop1, self.time_dependent_travel_time, self.time)
            backward_headway = behind_trip_arrival_time - self.time
            if backward_headway < 0.0:
                backward_headway = 0.0
        return backward_headway


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
        # trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        # last_arr_t = self.trip_log[trip_idx-1].stop_arr_times[STOPS_OUTBOUND[curr_stop_idx]]
        backward_headway = self.backward_headway()
        last_arr_t = self.stops[curr_stop_idx].last_arr_t
        if last_arr_t:
            forward_headway = self.time - last_arr_t[-2]
        else:
            print(f'what? last arrival time shows nan for controlled trip at time {self.time/60/60}')
            forward_headway = SCHED_DEP_IN[1] - SCHED_DEP_IN[0]
        print(f'forward {round(forward_headway/60, 2)} and backward {round(backward_headway/60, 2)}')
        # min_allowed_hw = self.control_strength * CONTROL_MEAN_HW
        # limit_holding = max(0, (last_arr_t + min_allowed_hw) - self.time)
        if stop.stop_id == STOPS_OUTBOUND[0]:
            # if backward_headway > forward_headway:
            #
            #     holding_time = min(limit_holding, (backward_headway - forward_headway)/2)
            #     if self.hold_adj_factor > 0.0 and holding_time > 0:
            #         holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
            #     assert holding_time >= 0
            #     self.outbound_dispatch(hold=holding_time)
            # else:
            #     self.outbound_dispatch()
            holding_time = min(LIMIT_HOLDING, max((backward_headway - forward_headway) / 2, 0))
            if self.hold_adj_factor > 0.0 and holding_time > 0:
                holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
            assert holding_time >= 0
            if self.bus.active_trip[0].trip_id in FOCUS_TRIPS:
                self.hold_ts = np.append(self.hold_ts, holding_time)
            self.outbound_dispatch(hold=holding_time)
        else:
            # self.fixed_stop_load()
            # if backward_headway > forward_headway:
            #     holding_time = min(limit_holding, (backward_headway - forward_headway)/2)
            #     if self.hold_adj_factor > 0.0 and holding_time > 0:
            #         holding_time = np.random.uniform(self.hold_adj_factor * holding_time, holding_time)
            #     assert holding_time >= 0
            #     self.fixed_stop_depart(hold=holding_time)
            # else:
            #     self.fixed_stop_depart()
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
        last_focus_bus = [bus for bus in self.buses if bus.bus_id == LAST_FOCUS_TRIP_BLOCK]
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in last_focus_bus[0].finished_trips]:
            return True

        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            # if not self.bus.finished_trips:
            #     # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
            #     self.smart_dispatch_initialized()
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
        self.bool_terminal_state = False
        self.pool_sars = []
        self.bus = Bus(0, [])

        # for records
        self.trajectories = {}
        self.trips_sars = {}
        for trip_id in TRIP_IDS_IN:
            self.trajectories[trip_id] = []
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

            interval_delay = get_interval(trip.sched_time, DELAY_INTERVAL_LENGTH_MINS) - DELAY_START_INTERVAL
            rand_percentile = np.random.uniform(0.0, 100.0)
            if trip.route_type == 0:
                dep_delay = np.percentile(DEP_DELAY_DIST_OUT[interval_delay], rand_percentile)
            elif trip.route_type == 1:
                dep_delay = np.percentile(DEP_DELAY1_DIST_IN[interval_delay], rand_percentile)
            else:
                assert trip.route_type == 2
                dep_delay = np.percentile(DEP_DELAY2_DIST_IN[interval_delay], rand_percentile)
            # random_delay = max(random.uniform(DEP_DELAY_FROM, DEP_DELAY_TO), 0)
            bus.next_event_time = trip.sched_time + max(dep_delay, 0)
            bus.dep_t = deepcopy(bus.next_event_time)
            bus.next_event_type = 3 if trip.route_type else 0
        # initialize passenger demand
        self.completed_pax = []
        self.initialize_pax_demand()
        return False

    def fixed_stop_load(self, skip=False):
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
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
        trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
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

        # if self.no_overtake_policy and trip_idx:
        #     prev_dep_t = self.trip_log[trip_idx-1].stop_dep_times[STOPS_OUTBOUND[curr_stop_idx]]
        #     if prev_dep_t is None:
        #         print('---report')
        #         print(f'trips in question {TRIP_IDS_IN[trip_idx - 1:trip_idx + 1]}')
        #         print(f'time {round(self.time)}')
        #         print(f'stop {curr_stop_idx}')
        #     bus.dep_t = max(prev_dep_t + NO_OVERTAKE_BUFFER, bus.dep_t)

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

        self.trip_log[trip_idx].stop_dep_times[STOPS_OUTBOUND[curr_stop_idx]] = bus.dep_t
        runtime = self.get_travel_time()
        bus.next_event_time = bus.dep_t + runtime

        # if self.no_overtake_policy:
        #     bus.next_event_time = deepcopy(self.no_overtake())
        bus.next_event_type = 2 if bus.next_stop_id == STOPS_OUTBOUND[-1] else 1
        self.record_trajectories(pickups=bus.ons, offs=bus.offs, denied_board=bus.denied, hold=hold, skip=skip)
        return

    def _add_observations(self):
        bus = self.bus
        curr_stop_idx = STOPS_OUTBOUND.index(bus.last_stop_id)
        # trip_idx = TRIP_IDS_IN.index(bus.active_trip[0].trip_id)
        # last_arr = self.trip_log[trip_idx-1].stop_arr_times[STOPS_OUTBOUND[curr_stop_idx]]
        # second_last_arr = self.trip_log[trip_idx-2].stop_arr_times[STOPS_OUTBOUND[curr_stop_idx]]
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

        t0_agent = self.trip_log[agent_trip_idx].stop_arr_times[s0]
        t1_agent = self.trip_log[agent_trip_idx].stop_arr_times[s1]

        t0_behind = self.trip_log[trip_idx].stop_arr_times[s0]
        t1_behind = self.trip_log[trip_idx].stop_arr_times[s1]

        for pax in self.completed_pax:
            if pax.trip_id == agent_trip_id and pax.orig_idx in stop_idx_set:
                pax_count += 1
                wait = pax.wait_time
                ride = min(pax.alight_time - pax.board_time, t1_agent - pax.board_time)
                assert ride > 0
                if pax.orig_idx == stop_idx_set[0] and pax.wait_time != 0.0:
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
                    wait = pax.wait_time
                    ride = t1_agent - pax.board_time
                    assert ride > 0
                    if pax.orig_idx == stop_idx_set[0] and pax.wait_time != 0.0:
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
                sum_rew_behind_wait_time += pax.wait_time
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
                sum_rew_behind_wait_time += pax.wait_time
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
            trip_idx = TRIP_IDS_IN.index(trip_id)
            neighbor_trip_id = TRIP_IDS_IN[trip_idx - 1]
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
                planned_fw_h = SCHED_DEP_IN[trip_idx] - SCHED_DEP_IN[trip_idx - 1]
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
        last_focus_bus = [bus for bus in self.buses if bus.bus_id == LAST_FOCUS_TRIP_BLOCK]
        if done or LAST_FOCUS_TRIP in [t.trip_id for t in last_focus_bus[0].finished_trips]:
            return True
        if self.bus.next_event_type == 0:
            # if the trip is initialized, check that it does not leave before the trip has departed
            # if there is a time conflict, then update its dispatch time to at least the next dispatch time
            # if not self.bus.finished_trips:
            #     # this function, if the previous trip hasn't departed, updates the dispatch time and returns False
            #     self.smart_dispatch_initialized()
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
