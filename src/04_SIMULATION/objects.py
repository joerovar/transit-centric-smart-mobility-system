import numpy as np
import pandas as pd
from copy import deepcopy

def td(t):
    return pd.to_timedelta(round(t), unit='S')


class Line:
    def __init__(self, line, stops, link_times, interval_length_mins,
                 outbound_direction, inbound_direction):
        self.id = line
        self.stops = {
            outbound_direction: stops[stops['direction']==outbound_direction].copy(), 
            inbound_direction: stops[stops['direction']==inbound_direction].copy()} 
        # Dataframe types, includes geo info, name
        # extract stop-to-stop times
        self.link_times = link_times
        self.link_t_interval_sec = int(interval_length_mins*60)
        self.interval_col_name = 'bin_' + str(interval_length_mins) + 'mins'
        self.hist_date = None # this will get updated by the user in reset fn
        
    def link_time_from_interval(self, stop_from, stop_to, 
                                time_sec, sample=False,
                                edge_link=False):
        link_times = self.link_times.copy()
        interval = int(time_sec/self.link_t_interval_sec)

        link_t = link_times.loc[
            (link_times[self.interval_col_name]==interval) &  
            (link_times['stop_id']==stop_from) & 
            (link_times['next_stop_id']==stop_to)]
        
        if link_t.empty:
            print(self.id, stop_from, stop_to)

        if edge_link:
            return link_t['schd_link_time'].values[0]*60

        if sample:
            return link_t['link_time'].sample().iloc[0] * 60
        else:
            return link_t['link_time'].mean() * 60
    
    def time_between_two_stops(self, stop0, stop1, stops, departs):
        t = 0 # travel time accumulator
        for i in range(stop0, stop1):
            first = True if i == 0 else False
            t += self.link_time_from_interval(stops[i], stops[i+1], 
                                              departs+t,
                                              edge_link=first)
        return t
    
    def link_time_from_trip(self, time_sec, trip, stop_from, stop_to, route_id,
                            edge_link=False):
        link_times = self.link_times.copy()
        
        # NEEDS A DATE TO BE PRE-LOADED
        link_time = link_times.loc[
            (link_times['trip_id']==trip) &
            (link_times['date']==self.hist_date) &
            (link_times['stop_id']==stop_from) & 
            (link_times['next_stop_id']==stop_to)]
        
        if link_time.empty:
            return self.link_time_from_interval(stop_from, stop_to, time_sec, route_id,
                                                edge_link=edge_link)
        elif edge_link:
            return link_time['schd_link_time'].values[0] * 60 # in case edge link (unreliable AVL)
        else:
            return link_time['link_time'].values[0] * 60
    
    def run_time_from_interval(self, direction, time_sec, sample=False):
        link_times = self.link_times.copy()
        interval = int(time_sec/self.link_t_interval_sec)

        # find trips that a) begin at the interval and b) are complete (all stop sequences)
        total_nr_stops = self.stops[direction].shape[0]
        times_grouped = link_times.groupby(
            ['date', 'trip_id'])['stop_sequence'].count().reset_index()
        times_grouped2 = link_times.groupby(['date', 'trip_id'])[self.interval_col_name].min().reset_index()
        times_grouped = times_grouped.merge(times_grouped2, on=['date', 'trip_id'])
        times_grouped = times_grouped.rename(columns={'stop_sequence':'total_stops',
                                                      self.interval_col_name:'min_bin'})
        # for that we groupcount by (date, trip, bin) and filter out the right bin and count
        link_times = link_times.merge(times_grouped, on=['date', 'trip_id'])
        link_t_focus = link_times.loc[(link_times['min_bin']==interval) &
                                     (link_times['total_stops']==total_nr_stops)].copy()
        run_t_focus = link_t_focus.groupby(
            ['date', 'trip_id'])['link_time'].sum().reset_index()
        if sample:
            return run_t_focus['link_time'].sample().iloc[0]
        else:
            return run_t_focus['link_time'].mean()

class Stop:
    def __init__(self, stop_id):
        self.stop_id = stop_id
        self.last_arr_time = None
        self.last_dep_time = None
        self.last_arr_trip = None
        self.last_dep_trip = None

class Route:
    def __init__(self):
        self.schedule = None
        self.vehicles = []
        self.stops = []

class FixedRoute(Route):
    def __init__(self, route, schedule, 
                 outbound_direction, inbound_direction, 
                 stops):
        super().__init__()
        self.id = route
        self.schedule = schedule.copy()
        self.schedule['departure_time_td'] = pd.to_timedelta(
            self.schedule['departure_time'])
        self.schedule = self.schedule.sort_values(
            by='departure_time_td').reset_index(drop=True)
        self.schedule['departure_time_sec'] = self.schedule['departure_time_td'].dt.total_seconds()
        self.outbound_direction = outbound_direction
        self.inbound_direction = inbound_direction
        self.stops_outbound = stops[stops['direction']==outbound_direction].copy()
        self.stops_inbound = stops[stops['direction']==inbound_direction].copy()
        self.stops = {
            outbound_direction: [Stop(si) for si in self.stops_outbound['stop_id'].tolist()],
            inbound_direction: [Stop(si) for si in self.stops_inbound['stop_id'].tolist()]
        }

    def update_schedule(self, confirmed_trips=None):
        # mark schedule for cancelled trips
        self.schedule['confirmed'] = 1
        if confirmed_trips:
            # if not all, type list with trip IDs
            self.schedule.loc[
                ~self.schedule['schd_trip_id'].isin(confirmed_trips), 'confirmed'] = 0
        return
    
    def reset_stop_records(self):
        self.stops = {
            self.outbound_direction: [Stop(si) for si in self.stops_outbound['stop_id'].tolist()],
            self.inbound_direction: [Stop(si) for si in self.stops_inbound['stop_id'].tolist()]
        }
    
    def next_terminal_departure(self, after, direction):
        schd = self.schedule.copy()
        next_schd = schd.loc[(schd['confirmed']==1) & 
                             (schd['departure_time_sec']>after) & 
                             (schd['stop_sequence']==1) &
                             (schd['direction']==direction)]
        return next_schd['departure_time_sec'].values[0]
    
    def find_next_trip(self, direction, 
                       curr_stop_idx, curr_trip_id,
                       min_dep_t):
        dep_t = None
        stop_idx = None
        stops = self.stops[direction]
        for i in range(curr_stop_idx-1, -1, -1):
            stop = stops[i]
            if stop.last_dep_trip != curr_trip_id:
                dep_t = stop.last_dep_time
                stop_idx = deepcopy(i)
                break
        if dep_t is None:
            schd_dep_t = self.get_terminal_departure(curr_trip_id)
            dep_t = self.next_terminal_departure(max(schd_dep_t, min_dep_t),
                                                 direction)
            stop_idx = 0
        return dep_t, stop_idx
    
    def get_terminal_departure(self, trip_id):
        schd = self.schedule.copy()
        trip_schd = schd[(schd['schd_trip_id']==trip_id) & 
                         (schd['stop_sequence']==1)].copy()
        return trip_schd['departure_time_sec'].values[0]

class Demand:
    def __init__(self, od):
        self.od = od
        self.pax_at_stops = None
        self.pax_served = None
    
    def generate(self, interval_length_mins, start_time_sec, end_time_sec):
        pax_at_stops = {
            'boarding_stop': [],
            'alighting_stop': [],
            'arrival_time': [],
            'times_denied': [],
            'route_id': []
        }

        bin_col_name = 'bin_' + str(interval_length_mins) + 'mins'

        interval = int(start_time_sec/interval_length_mins/60)
        end_interval = int(end_time_sec/interval_length_mins/60)

        bins_od = self.od[self.od[bin_col_name].isin(range(interval, end_interval+1))].copy()
        bins_od = bins_od.sort_values(by=bin_col_name).reset_index(drop=True)
        bins_od['avg_wait'] = 60*interval_length_mins/bins_od['pax']

        avg_arrival_time = np.array(bins_od['avg_wait'])[:,np.newaxis]
        max_nr_pax = int(
            2.5*np.round(interval_length_mins*60/avg_arrival_time.min())) 
        # based on the smallest avg wait
        sampled_arrival_times = np.random.exponential(
            avg_arrival_time, size=(avg_arrival_time.size,max_nr_pax))
        cu_samp_arrival_times = np.cumsum(sampled_arrival_times, axis=1)

        stop_cols_od = bins_od[['boarding_stop', 'alighting_stop', 
                                bin_col_name, 'route_id']].copy()

        for i in stop_cols_od.index:
            arr_t_tmp = cu_samp_arrival_times[i]
            arr_t_valid = arr_t_tmp[arr_t_tmp<=interval_length_mins*60] 
            # all arrival times less than the bound of the interval
            if arr_t_valid.size:
                tmp_stop_cols = stop_cols_od.loc[i]
                arr_t_valid += tmp_stop_cols[bin_col_name] * interval_length_mins * 60
                pax_at_stops['boarding_stop'] += [tmp_stop_cols['boarding_stop']] * arr_t_valid.size
                pax_at_stops['alighting_stop'] += [tmp_stop_cols['alighting_stop']] * arr_t_valid.size
                pax_at_stops['arrival_time'] += list(arr_t_valid)
                pax_at_stops['times_denied'] += [0] * arr_t_valid.size
                pax_at_stops['route_id'] += [tmp_stop_cols['route_id']] * arr_t_valid.size

        self.pax_at_stops = pd.DataFrame(pax_at_stops)
        self.pax_served = pd.DataFrame()
    
    def get_pax_to_board(self, time, stop, capacity_avail, route_id,
                         remaining_stops):
        pax_at_stops = self.pax_at_stops.copy()

        valid_pax = pax_at_stops[(pax_at_stops['boarding_stop']==stop) & 
                                (pax_at_stops['arrival_time']<=time) &
                                (pax_at_stops['route_id']==route_id) &
                                (pax_at_stops['alighting_stop'].isin(remaining_stops))].copy()

        if valid_pax.empty:
            return valid_pax
        else:
            if capacity_avail < valid_pax.shape[0]:
                pax_at_stops.loc[
                    valid_pax.iloc[capacity_avail:].index, 'times_denied'] += 1
                valid_pax = valid_pax.iloc[:capacity_avail]
            pax_at_stops = pax_at_stops.drop(valid_pax.index)
            self.pax_at_stops = pax_at_stops.copy()        
            return valid_pax
    
    def record_pax_served(self, pax_served):
        self.pax_served = pd.concat(
            [self.pax_served, pax_served]).reset_index(drop=True)
        return
    

class Trip:
    def __init__(self, trip_schd):
        self.id = trip_schd['schd_trip_id'].values[0]
        self.route_id = trip_schd['route_id'].values[0]
        self.direction = trip_schd['direction'].values[0]
        self.schedule = trip_schd.copy()
        self.stops = trip_schd['stop_id'].tolist()

        self.records = {
            'stop_id': [],
            'stop_sequence': [],
            'arrival_time_sec': [],
            'departure_time_sec': [],
            'passenger_load': [],
            'ons': [],
            'offs': [],
            'schd_time_sec': []
        }
    
    def record_arrival(self, stop_idx, time_sec, ons, offs):
        stop_schd_info = self.schedule.loc[stop_idx]
        self.records['stop_id'].append(
            stop_schd_info['stop_id'])
        self.records['schd_time_sec'].append(
            stop_schd_info['departure_time_sec'])
        self.records['arrival_time_sec'].append(time_sec)
        self.records['ons'].append(ons)
        self.records['offs'].append(offs)
        self.records['stop_sequence'].append(stop_idx+1)

    def record_departure(self, time_sec, ons, pax_load):
        self.records['departure_time_sec'].append(time_sec)
        self.records['ons'][-1] += ons
        self.records['passenger_load'].append(pax_load)

    def get_complete_records(self):
        df_rec = pd.DataFrame(self.records)
        df_rec['trip_id'] = self.id
        df_rec['block_id'] = self.schedule['block_id'].values[0]
        df_rec['run_id'] = self.schedule['runid'].values[0]
        df_rec['direction'] = self.direction
        df_rec['route_id'] = self.route_id
        return df_rec


class Vehicle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pax = pd.DataFrame()
        self.status = None
        self.status_dict = {}
        self.next_event = None
        self.event_dict = {}


class FixedVehicle(Vehicle):
    def __init__(self, capacity, block_schedule, dwell_time_params, 
                 report_delays, route_id, break_threshold, max_early_dev):
        super().__init__(capacity)
        self.route_id = route_id
        self.block_id = int(block_schedule['block_id'].iloc[0])
        self.past_trips = []
        self.next_trips = []
        self.curr_trip = None
        self.dwell_time_params = dwell_time_params
        self.break_threshold = break_threshold
        self.max_early_dev = max_early_dev
        self.trip_records = pd.DataFrame()

        # set status
        self.status_dict = {
            -1: 'inactive - finished all trips',
            0: 'inactive - yet to report for first trip',
            1: 'inactive - terminal turnaround',
            2: 'active - dwell at stops',
            3: 'active - between stops',
            4: 'inactive - reported for first trip',
            5: 'inactive - break between shifts'
        }
        self.status = 0
        self.stop_idx = 0
        
        # assign trips 
        for t in block_schedule['trip_id'].unique():
            trip_sched = block_schedule[block_schedule['trip_id']==t].copy()
            trip_sched = trip_sched.sort_values(
                by='departure_time_sec').reset_index(drop=True)
            self.next_trips.append(Trip(trip_sched))
        
        self.event_dict = {
            0: 'terminal boarding',
            1: 'arrive at stop',
            2: 'depart stop or terminal',
            3: 'arrive at terminal',
            4: 'first report at terminal',
            5: 'report after break'
        }

        first_schd_time = self.next_trips[0].schedule['departure_time_sec'].values[0]
        self.report_delay = np.random.uniform(*report_delays)*60 # convert to seconds
        # print(f'BLOCK ------ {self.block_id}')
        # print(pd.to_timedelta(round(first_schd_time), unit='S'))
        # print(pd.to_timedelta(round(report_delay), unit='S'))
        # print(pd.to_timedelta(round(first_schd_time+report_delay), unit='S'))

        self.next_event = {
            't': first_schd_time + self.report_delay,
            'type': 4
        }
        self.last_event = {
            't': None
        }

    def _board_pax(self, time, pax2board):
        pax2board['boarding_time'] = time
        pax2board['trip_id'] = self.curr_trip.id
        pax2board['direction'] = self.curr_trip.direction
        pax2board['alighting_time'] = 0.0
        self.pax = pd.concat([self.pax, pax2board]).reset_index(drop=True)
        return
    
    def _dwell_time(self, boardings, alightings):
        board_time = boardings * self.dwell_time_params['board']
        alight_time = alightings * self.dwell_time_params['alight']
        acc_dec = self.dwell_time_params['acc_dec'] if boardings+alightings else 0
        error_term = np.random.uniform(
            0,self.dwell_time_params['error']) if boardings+alightings else 0
        return board_time + alight_time + acc_dec + error_term
    
    def _alight_pax(self, time):
        stop = self.curr_trip.stops[self.stop_idx]
        pax = self.pax.copy()
        alight_pax = pax[pax['alighting_stop']==stop].copy()
        if not alight_pax.empty:
            pax = pax.drop(alight_pax.index).reset_index(drop=True)
            alight_pax['alighting_time'] = time
            self.pax = pax.copy()
        return alight_pax

    def terminal_boarding(self, time, demand, route):
        self.curr_trip = self.next_trips.pop(0)
        self.stop_idx = 0

        # pax 
        capacity_avail = self.capacity - self.pax.shape[0]
        stop = self.curr_trip.stops[0]
        remaining_stops = self.curr_trip.stops[1:]
        pax2board = demand.get_pax_to_board(time, stop, 
                                            capacity_avail, self.route_id,
                                            remaining_stops)
        if not pax2board.empty:
            self._board_pax(time, pax2board)
        
        self.curr_trip.record_arrival(0, time, pax2board.shape[0], 0) # record step

        # update stop
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_time = time
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_trip = self.curr_trip.id

        # updates status and next event information
        self.status = 2
        self.next_event['type'] = 2
        self.next_event['t'] = time + self._dwell_time(pax2board.shape[0], 0)
        return

    def depart_stop(self, time, line, demand, route):
        # if there was dwell time allow more pax to board
        capacity_avail = self.capacity - self.pax.shape[0]
        stop = self.curr_trip.stops[self.stop_idx]
        remaining_stops = self.curr_trip.stops[self.stop_idx+1:]
        pax2board = demand.get_pax_to_board(time, stop, 
                                            capacity_avail, self.route_id,
                                            remaining_stops)
        if not pax2board.empty:
            self._board_pax(time, pax2board)

        # if it's the first or last link -> link time must be the scheduled
        nr_stops = len(self.curr_trip.stops)
        first = True if self.stop_idx == 0 else False
        last = True if self.stop_idx + 1 == nr_stops-1 else False

        stop0 = self.curr_trip.stops[self.stop_idx]
        stop1 = self.curr_trip.stops[self.stop_idx+1]
        link_time = line.link_time_from_trip(
            time, self.curr_trip.id, stop0, stop1, self.route_id, 
            edge_link=(first or last))
        
        # update stop
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_dep_time = time
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_dep_trip = self.curr_trip.id
        
        self.curr_trip.record_departure(time, pax2board.shape[0], self.pax.shape[0]) # record
        
        # update status and next event information
        self.status = 3 
        self.next_event['t'] = time + link_time
        self.next_event['type'] = 3 if last else 1 # arrival at terminal or at stop

        return

    def adjust_departure(self, adjustment_mins=None, equal_headways=False):
        if adjustment_mins: # can be negative for depart early
            return
        if equal_headways:
            return

    def arrive_at_stop(self, time, demand, route):
        self.stop_idx += 1
        
        # update pax
        capacity_avail = self.capacity - self.pax.shape[0]
        stop = self.curr_trip.stops[self.stop_idx]
        remaining_stops = self.curr_trip.stops[self.stop_idx+1:]
        pax2board = demand.get_pax_to_board(time, stop, 
                                            capacity_avail, self.route_id,
                                            remaining_stops)
        if not pax2board.empty:
            self._board_pax(time, pax2board)

        if not self.pax.empty:
            pax2alight = self._alight_pax(time)
            if not pax2alight.empty:
                demand.record_pax_served(pax2alight)
        else:
            pax2alight = pd.DataFrame()

        # update stop
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_time = time
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_trip = self.curr_trip.id

        # record step
        self.curr_trip.record_arrival(self.stop_idx, time, 
                                      pax2board.shape[0], pax2alight.shape[0])

        # update status and next event
        self.status = 2
        self.next_event['t'] = time + self._dwell_time(pax2board.shape[0],
                                                       pax2alight.shape[0])
        self.next_event['type'] = 2
        return
    
    def arrive_at_terminal(self, time, demand, route):
        self.stop_idx += 1

        # update pax
        if not self.pax.empty:
            pax2alight = self._alight_pax(time)
            if not pax2alight.empty:
                demand.record_pax_served(pax2alight)
        else:
            pax2alight = pd.DataFrame()
        
        if self.pax.shape[0]:
            print(f'load at terminal not zero instead {self.pax.shape[0]}')
            print(self.pax.head()) # load at terminal should be zero

        self.curr_trip.record_arrival(self.stop_idx, time,
                                      0, pax2alight.shape[0])
        self.curr_trip.record_departure(time, 0, 0)

        # update stop
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_time = time
        route.stops[
            self.curr_trip.direction][self.stop_idx].last_arr_trip = self.curr_trip.id

        # record step     
        trip_record = self.curr_trip.get_complete_records() 
        self.trip_records = pd.concat(
            [self.trip_records, trip_record]).reset_index(drop=True)

        # update
        self.past_trips.append(deepcopy(self.curr_trip))
        self.curr_trip = None

        # update status
        if self.next_trips:
            schd_time = self.next_trips[0].schedule['departure_time_sec'].values[0]
            if schd_time - time > self.break_threshold * 60:
                self.status = 5
                self.next_event['type'] = 5
                self.next_event['t'] = schd_time - self.max_early_dev*60 # TO-DO randomize at some point!
            else:
                self.status = 1
                self.next_event['type'] = 0
                self.next_event['t'] = max(time + self._dwell_time(0, pax2alight.shape[0]), schd_time)

        else:
            self.status = -1
            self.next_event['t'] = -1
            self.next_event['type'] = -1
        return
    
    def report_at_terminal(self, time):
        self.status = 4
        schd_time = self.next_trips[0].schedule['departure_time_sec'].values[0]
        self.next_event['t'] = max(time, schd_time)
        self.next_event['type'] = 0
        return
    
    def process_event(self, time, demand, line, route):
        self.last_event['t'] = deepcopy(time)
        if self.next_event['type'] == 0:
            self.terminal_boarding(time, demand, route)
            return
        if self.next_event['type'] == 1:
            self.arrive_at_stop(time, demand, route)
            return
        if self.next_event['type'] == 2:
            self.depart_stop(time, line, demand, route)
            return
        if self.next_event['type'] == 3:
            self.arrive_at_terminal(time, demand, route)
            return
        if self.next_event['type'] in [4, 5]:
            self.report_at_terminal(time)
            return

    def compute_headways(self, line, min_dep_t, route, terminal=False):
        # pre headway
        trip = self.next_trips[0] if terminal else self.curr_trip
        stop = route.stops[trip.direction][self.stop_idx]
        if stop.last_dep_time is None:
            return None, None
        last_dep_t = stop.last_dep_time
        min_dep_t = max(min_dep_t, last_dep_t)
        pre_hw = min_dep_t - last_dep_t
        if pre_hw == 0.0:
                print('saved a negative headway')
                print(f'earliest {td(min_dep_t)} last {td(last_dep_t)} true earliest {td(max(min_dep_t, last_dep_t))}')
        
        # next headway
        # here we try to find the last stop that the next trip departed from
        # next trip details
        dep_t, stop_idx_from = route.find_next_trip(trip.direction,self.stop_idx, 
                                                    trip.id, min_dep_t)

        travel_time = line.time_between_two_stops(stop_idx_from, self.stop_idx, 
                                                  dep_t, trip.stops)
        next_arrives = dep_t + travel_time
        next_hw = next_arrives - min_dep_t
        
        # print(f'from terminal or stop {stop_idx_from} to stop {self.stop_idx}')
        # print(f'departure {td(dep_t)}')
        # print(f'travel time {td(travel_time)}')
        return pre_hw, next_hw

    def get_info(self, line, time):
        if self.status in [2,3]:
            # only active vehicles here
            stops = line.stops[self.curr_trip.direction]
            last_stop_id = self.curr_trip.stops[self.stop_idx]
            last_stop = stops.loc[stops['stop_id']==last_stop_id].copy()
            stop_lat, stop_lon = last_stop[['stop_lat', 'stop_lon']].values[0].tolist()
            message = {
                'time': time,
                'route_id': self.route_id,
                'id': self.block_id,
                'status': self.status,
                'status_desc': self.status_dict[self.status],
                'next_event': self.event_dict[self.next_event['type']],
                'next_event_t': self.next_event['t'],
                't_until_next': self.next_event['t'] - time,
                'stop_id': last_stop_id,
                'stop_sequence': self.stop_idx+1,
                'stop_lat': stop_lat,
                'stop_lon': stop_lon, 
                'direction': self.curr_trip.direction,
                'trip_id': self.curr_trip.id,
                'active': 1, 
                't_since_last': time - self.last_event['t'],
                'pax_load': self.pax.shape[0],
                'trip_sequence': self.curr_trip.schedule['trip_sequence'].iloc[0]
            }
            return message
        if self.status in [0,1,4,5]:
            # trips awaiting departure
            message = {
                'time': time,
                'route_id': self.route_id,
                'id': self.block_id,
                'status': self.status,
                'status_desc': self.status_dict[self.status],
                'next_event': self.event_dict[self.next_event['type']],
                'next_event_t': self.next_event['t'],
                't_until_next': self.next_event['t'] - time,
                'stop_id': self.next_trips[0].stops[0],
                'stop_sequence': 1,
                'stop_lat': 0,
                'stop_lon': 0,
                'direction': self.next_trips[0].direction,
                'trip_id': self.next_trips[0].id,
                'active': 0,
                't_since_last': self.last_event['t'] if self.last_event['t'] is None else time - self.last_event['t'],
                'pax_load': self.pax.shape[0],
                'trip_sequence': self.next_trips[0].schedule['trip_sequence'].iloc[0]
            }
            return message
        if self.status == -1:
            # trips awaiting departure
            message = {
                'time': time,
                'route_id': self.route_id,
                'id': self.block_id,
                'status': self.status,
                'status_desc': self.status_dict[self.status],
                'next_event': None,
                'next_event_t': None,
                't_until_next': None,
                'stop_id': None,
                'stop_sequence': None,
                'stop_lat': 0,
                'stop_lon': 0,
                'direction': None,
                'trip_id': None,
                'active': 0,
                't_since_last': self.last_event['t'] if self.last_event['t'] is None else time - self.last_event['t'],
                'pax_load': None,
                'trip_sequence': None
            }
            return message
        return None

class FlexVehicle(Vehicle):
    def __init__(self, capacity):
        super().__init__(capacity)



