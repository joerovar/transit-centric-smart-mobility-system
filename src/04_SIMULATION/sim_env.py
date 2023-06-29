from objects import Line, FixedVehicle, FixedRoute, Demand
import pandas as pd
import numpy as np
from constants import *
from copy import deepcopy

def td(t):
    return pd.to_timedelta(t, unit='S')

def formatTime(t):
    return t.dt.strftime('%H:%M:%S.%f').str[:-5]

def flag_holding_event(info, direction, stop_seq):
    tmp_info = info[info['t_since_last'].notna()].copy()
    flag_vehicle = tmp_info[(tmp_info['direction']==direction) & 
                            (tmp_info['stop_sequence'].isin(stop_seq)) & 
                            (tmp_info['status']==2) &
                            (tmp_info['t_since_last']==pd.to_timedelta(0, unit='S'))].copy()
    return flag_vehicle
    
def flag_departure_event(info):
    tmp_info = info[info['t_since_last'].notna()].copy()
    outbound_terminals = [OUTBOUND_TERMINALS[rt][0] for rt in ROUTES]
    flag_vehicle = tmp_info[(tmp_info['stop_id'].isin(outbound_terminals)) & 
                            (tmp_info['status'] == 4) & 
                            (tmp_info['t_since_last']==pd.to_timedelta(0, unit='S'))].copy()
    return flag_vehicle

def recommended_dep_t(pre_hw, next_hw, t, max_dep_t):
    hw_diff = next_hw-pre_hw
    hold_wo_lim = max(0,hw_diff/2)
    rec_wo_lim = t + hold_wo_lim # without limit
    rec_dep_t = min(rec_wo_lim, max_dep_t)
    true_hold = rec_dep_t - t
    new_hws = pre_hw+true_hold, next_hw-true_hold
    return rec_dep_t, new_hws, rec_dep_t

def get_layover_bus_dep(info, control_vehs, schd):
    rt_id = control_vehs['route_id'].iloc[0]
    stop_id = control_vehs['stop_id'].iloc[0]
    trip_seq = control_vehs['trip_sequence'].iloc[0]
    
    # the last trip confirmed
    last_trip = schd[(schd['stop_sequence']==1) &
                     (schd['stop_id']==stop_id) &
                     (schd['trip_sequence'] < trip_seq) &
                     (schd['confirmed']==1)].copy()
    if last_trip.empty:
        return None
    last_trip_id = last_trip['schd_trip_id'].iloc[-1]

    lay_bus = info[(info['route_id']==rt_id) & 
                (info['stop_id']==stop_id) &
                (info['stop_sequence']==1) & 
                (info['trip_id'] == last_trip_id) &
                (info['status'].isin([1,2,4]))].copy()
    if lay_bus.empty:
        return None
    assert lay_bus.shape[0]==1
    # these are buses either in dwell or that reported and have an assigned departure
    if lay_bus['status'].iloc[0] in (2,4):
        return lay_bus['next_event_t'].iloc[0].total_seconds()
    else:
        assert lay_bus['status'].iloc[0] == 1
        lay_bus_schd_dep = last_trip['departure_sec'].iloc[-1]
        return max(lay_bus_schd_dep, lay_bus['next_event_t'].iloc[0].total_seconds())

class SimEnv:
    def __init__(self):
        self.start_time_sec = pd.to_timedelta(START_TIME).total_seconds()
        self.end_time_sec = pd.to_timedelta(END_TIME).total_seconds()

        self.time = None
        self.stops = []
        self.vehicles = []

        self.step_counter = 0

    def reset(self):
        self.time = self.start_time_sec
        self.vehicles = []
        self.step_counter = 0
        return

    def step(self):
        return

class FixedSimEnv(SimEnv):
    def __init__(self):
        super(FixedSimEnv, self).__init__()
        stops = pd.read_csv(SIM_INPUTS_PATH + 'stops.csv')
        self.link_times = pd.read_csv(SIM_INPUTS_PATH + 'link_times.csv')
        self.hist_date = None
        schedule = pd.read_csv(SIM_INPUTS_PATH + 'schedule.csv')
        od = pd.read_csv(SIM_INPUTS_PATH + 'od.csv')

        for df in (stops, self.link_times, schedule, od):
            df['route_id'] = df['route_id'].astype(str)
        self.routes = {}
        self.lines = {}
        for route in ROUTES:
            rt_stops = stops[stops['route_id']==route].copy()
            rt_link_times = self.link_times[
                self.link_times['route_id']==route].copy()
            rt_schd = schedule[schedule['route_id']==route].copy()

            self.lines[route] = Line(
                route, rt_stops, rt_link_times, INTERVAL_LENGTH_MINS,
                OUTBOUND_DIRECTIONS[route], INBOUND_DIRECTIONS[route]) 
            self.routes[route] = FixedRoute(route, rt_schd, 
                                            OUTBOUND_DIRECTIONS[route],
                                            INBOUND_DIRECTIONS[route], rt_stops)
        self.demand = Demand(od)

        self.info_records = []

    def _update_info(self):
        # get information on vehicles that indicates time until next event
        info_vehicles = []
        for v in self.vehicles:
            v_status = v.get_info(self.lines[v.route_id], self.time)
            if v_status is not None:
                info_vehicles.append(v_status)
        df_info = pd.DataFrame.from_records(info_vehicles)
        df_info['nr_step'] = deepcopy(self.step_counter)
        df_info['date'] = self.hist_date
        self.info_records.append(df_info)
        return
    
    def _display_info(self, info=None):
        # display info 
        df_disp = self.info_records[-1].copy() # if not specified, just grab latest
        
        if info is not None:
            df_disp = info.copy()
        
        df_disp = df_disp.sort_values(by='t_until_next')
        df_obs = df_disp.copy()

        for col in ('time', 't_until_next', 'next_event_t', 't_since_last'):
            df_obs[col] = td(df_obs[col])

        df_disp['date'] = pd.to_datetime(df_disp['date'])

        df_disp['time'] = df_disp['date'] + td(df_disp['time'].round(decimals=1))
        df_disp['time'] = formatTime(df_disp['time'])

        df_disp['next_event_t'] = df_disp['date'] + td(df_disp['next_event_t'].round(decimals=1))
        df_disp['next_event_t'] = formatTime(df_disp['next_event_t'])

        df_disp['t_until_next'] = df_disp['t_until_next'].round(decimals=1)
        df_disp['t_since_last'] = df_disp['t_since_last'].round(decimals=1)

        disp_cols = ['time', 'nr_step', 'route_id', 'block_id', 'direction',
                     'active', 'status', 'status_desc', 
                     'next_event', 'next_event_t', 't_until_next', 
                     'stop_id','stop_sequence', 'pax_load', 't_since_last', 
                     'trip_id', 'trip_sequence']
        return df_disp[disp_cols], df_obs[disp_cols]
    
    def get_trip_records(self, scenario=None):
        trip_records = []
        for veh in self.vehicles:
            if not veh.trip_records.empty:
                trip_records.append(veh.trip_records)
        df_trips = pd.concat(trip_records, ignore_index=True)
        hist_date_dt = pd.to_datetime(self.hist_date)
        for ky in ('arrival', 'departure', 'schd'):
            full_ky = ky + '_sec'
            df_trips[full_ky] = pd.to_timedelta(
                df_trips[full_ky].round(), unit='S')
            df_trips[full_ky] += hist_date_dt
            df_trips.rename(columns={full_ky: ky + '_time'})
        if scenario:
            df_trips['scenario'] = scenario
        return df_trips
    
    def get_pax_records(self, scenario=None):
        pax_served = self.demand.pax_served.copy()
        hist_date_dt = pd.to_datetime(self.hist_date)
        for ky in ('arrival', 'alighting', 'boarding'):
            full_ky = ky + '_time'
            pax_served[full_ky] = pd.to_timedelta(
                pax_served[full_ky].round(), unit='S')
            pax_served[full_ky] += hist_date_dt
        if scenario:
            pax_served['scenario'] = scenario
        return pax_served

    def reset(self, hist_date=None):
        super().reset()
        self.info_records = []
        self.hist_date = hist_date

        if hist_date is None:
            raise ImportError

        for rt in ROUTES:
            self.lines[rt].hist_date = hist_date
            confirmed_trips = self.lines[rt].link_times.loc[
                self.lines[rt].link_times['date']==hist_date, 'trip_id'].tolist() # what was observed
            self.routes[rt].update_schedule(confirmed_trips=confirmed_trips) 
            self.routes[rt].reset_stop_records()

        self.demand.generate(INTERVAL_LENGTH_MINS, self.start_time_sec, 
                             self.end_time_sec)
        for rt in ROUTES:
            schedule = self.routes[rt].schedule.copy()
            blocks = schedule['block_id'].unique().tolist()

            for b in blocks: # Blocks may contain cancelled trips (based on runs)
                block_schedule = schedule[(schedule['block_id']==b) & 
                                        (schedule['confirmed']==1)].copy()
                if not block_schedule.empty:
                    vehicle = FixedVehicle(CAPACITY, block_schedule, 
                                           DWELL_TIME_PARAMS, TERMINAL_REPORT_DELAYS,
                                           rt, BREAK_THRESHOLD, MAX_EARLY_DEV)
                    self.vehicles.append(vehicle)

        
        # process first event
        self.step_counter += 1
        self._update_info()

        idx_next = self.info_records[-1][['t_until_next']].idxmin().values[0]
        next_vehicle = self.vehicles[idx_next]
        self.time = deepcopy(next_vehicle.next_event['t'])
        route = next_vehicle.route_id
        next_vehicle.process_event(self.time, self.demand, 
                                   self.lines[route], self.routes[route])

        self.step_counter += 1
        self._update_info()

        info = self._display_info()

        return [], None, 0, info
    
    def step(self):
        if self.time >= self.end_time_sec:
            return [], None, 1, {}
        
        idx_next = self.info_records[-1][['t_until_next']].idxmin().values[0]
        next_vehicle = self.vehicles[idx_next]
        self.time = deepcopy(next_vehicle.next_event['t'])
        route = next_vehicle.route_id
        next_vehicle.process_event(self.time, self.demand, 
                                   self.lines[route], self.routes[route])

        self.step_counter += 1

        self._update_info()

        info = self._display_info()
        
        return [], None, 0, info
    
    def get_headway(self, info, control_vehs, min_dep_t, terminal=False):
        control_veh = self.vehicles[control_vehs.index[0]]
        trip = control_veh.next_trips[0] if terminal else control_veh.curr_trip
        expected_dep = max(self.time, min_dep_t)
        route = self.routes[control_veh.route_id]

        if terminal: # adjust departure to schedule or possible layover
            # schd_dep_t = trip.schedule['departure_sec'].values[0]
            # expected_dep = max(expected_dep, schd_dep_t)
            layover_bus_dep = get_layover_bus_dep(info, control_vehs, 
                                                  route.schedule)
            if layover_bus_dep:
                expected_dep = max(expected_dep, layover_bus_dep)
                hw = expected_dep - layover_bus_dep
                return hw, expected_dep

        stop = route.stops[trip.direction][control_veh.stop_idx]
        if not stop.last['departure_time']:
            return None, None
        last_dep = stop.last['departure_time'][-1]
        hw = expected_dep - last_dep
        return hw, expected_dep
    
    def get_next_headway(self, control_vehs, terminal=False,
                         expected_dep=0):
        control_veh = self.vehicles[control_vehs.index[0]]
        trip = control_veh.next_trips[0] if terminal else control_veh.curr_trip
        expected_dep = max(self.time, expected_dep)

        route = self.routes[control_veh.route_id]
        dep_t, stop_idx_from = route.find_next_trip(trip,control_veh.stop_idx)

        line = self.lines[control_veh.route_id]
        travel_time = line.time_between_two_stops(stop_idx_from, control_veh.stop_idx, 
                                                  dep_t, trip.stops)
        next_arrives = dep_t + travel_time
        hw = next_arrives - expected_dep        
        return hw
    
    def terminal_dep_limits(self, route_id, trip_id):
        route = self.routes[route_id]
        schd_dep_t = route.get_terminal_departure(trip_id)
        earliest = max(schd_dep_t-MAX_EARLY_DEV*60, self.time)
        latest = max(schd_dep_t+MAX_LATE_DEV*60, self.time)
        return earliest, latest
    
    def adjust_departure(self, control_vehs, new_dep_t):
        control_veh = self.vehicles[control_vehs.index[0]]
        control_veh.next_event['t'] = deepcopy(new_dep_t)
        idx = control_vehs.index[0]
        info_record = self.info_records[-1]
        info_record.loc[idx, 'next_event_t'] = new_dep_t
        info_record.loc[idx, 't_until_next'] = new_dep_t - self.time
        updated_info = self._display_info()
        return updated_info
    
    def adjust_route_and_departure(self, control_vehs, new_dep_t):
        control_veh = self.vehicles[control_vehs.index[0]]
        control_veh.next_event['t'] = deepcopy(new_dep_t)
        idx = control_vehs.index[0]
        info_record = self.info_records[-1]
        info_record.loc[idx, 'next_event_t'] = new_dep_t
        info_record.loc[idx, 't_until_next'] = new_dep_t - self.time
        info_record.loc[idx, 'route_id'] = control_veh.route_id
        info_record.loc[idx, 'stop_id'] = control_veh.next_trips[0].stops[0]
        info_record.loc[idx, 'direction'] = control_veh.next_trips[0].direction
        info_record.loc[idx, 'trip_id'] = control_veh.next_trips[0].id
        info_record.loc[idx, 'trip_sequence'] = control_veh.next_trips[0].schedule['trip_sequence'].iloc[0]
        updated_info = self._display_info()
        return updated_info

class FlexSimEnv(SimEnv):
    """"""

    def reset(self):
        super().__init__()
