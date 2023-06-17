from objects import Line, FixedVehicle, FixedRoute, Demand
import pandas as pd
import numpy as np
from constants import *
from copy import deepcopy


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
                            (tmp_info['status'].isin([1,4])) & 
                            (tmp_info['t_since_last']==pd.to_timedelta(0, unit='S'))].copy()
    return flag_vehicle

def relevant_schedule(routes, time, out_terminals):
    # the point is to show from the schedule 
    # the previous 2 trips and the next 2 trips for each route
    lst_schds = []
    for rt_id in routes.keys():
        df = routes[rt_id].schedule.copy()
        df = df[(df['stop_id']==out_terminals[rt_id][0]) &
                (df['stop_sequence']==1)].reset_index(drop=True)
        curr_idx = df[df['departure_time_sec']>time].index[0]
        sub_cols = ['route_id', 'confirmed','runid', 'schd_trip_id', 
                    'departure_time', 'departure_time_sec', 
                    'trip_sequence', 'block_id']
        idx_from = max(0, curr_idx-2)
        idx_to = min(df.shape[0], curr_idx+3)
        lst_schds.append(df.loc[idx_from:idx_to, sub_cols])
    return pd.concat(lst_schds, ignore_index=True)

def recommended_dep_t(pre_hw, next_hw, t):
    hw_diff = next_hw-pre_hw
    rec_dep_t = t + max(0,hw_diff/2)
    return rec_dep_t

def get_layover_bus_dep(info, control_vehs):
    rt_id = control_vehs['route_id'].iloc[0]
    stop_id = control_vehs['stop_id'].iloc[0]
    trip_seq = control_vehs['trip_sequence'].iloc[0]
    lay_buses = info[(info['route_id']==rt_id) & 
                (info['stop_id']==stop_id) & 
                (info['status'].isin([1,2,4])) & 
                (info['stop_sequence']==1) & 
                (info['trip_sequence'] < trip_seq)].copy()
    if lay_buses.empty:
        return None
    return lay_buses['next_event_t'].max().total_seconds()

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

    def _get_info_vehicles(self):
        # get information on vehicles that indicates time until next event
        info_vehicles = []
        for v in self.vehicles:
            v_status = v.get_info(self.lines[v.route_id], self.time)
            if v_status is not None:
                info_vehicles.append(v_status)
        df_info = pd.DataFrame.from_records(info_vehicles)
        return df_info
    
    def _display_info(self, info=None):
        # display info 
        df_disp = self.info_records[-1].copy() # if not specified, just grab latest

        if info is not None:
            df_disp = info.copy()

        df_disp = df_disp.sort_values(by='t_until_next')
        df_disp['time'] = pd.to_timedelta(df_disp['time'].round(decimals=1), unit='S')
        df_disp['t_until_next'] = pd.to_timedelta(df_disp['t_until_next'].round(decimals=1), unit='S')
        df_disp['next_event_t'] = pd.to_timedelta(df_disp['next_event_t'].round(decimals=1), unit='S')
        df_disp['t_since_last'] = pd.to_timedelta(df_disp['t_since_last'].round(decimals=1), unit='S')
        disp_cols = ['time', 'nr_step', 'route_id', 'id','active', 'status', 'status_desc', 
         'next_event', 'next_event_t', 't_until_next', 'stop_id','stop_sequence', 
         'direction', 'pax_load', 't_since_last', 'trip_id', 'trip_sequence']
        return df_disp[disp_cols]
    
    def get_trip_records(self, scenario=None):
        trip_records = []
        for veh in self.vehicles:
            if not veh.trip_records.empty:
                trip_records.append(veh.trip_records)
        df_trips = pd.concat(trip_records, ignore_index=True)
        hist_date_dt = pd.to_datetime(self.hist_date)
        for ky in ('arrival', 'departure', 'schd'):
            full_ky = ky + '_time_sec'
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
        df_info = self._get_info_vehicles()
        df_info['nr_step'] = deepcopy(self.step_counter)
        self.info_records.append(df_info)

        idx_next = self.info_records[-1][['t_until_next']].idxmin().values[0]
        next_vehicle = self.vehicles[idx_next]
        self.time = deepcopy(next_vehicle.next_event['t'])
        route = next_vehicle.route_id
        next_vehicle.process_event(self.time, self.demand, 
                                   self.lines[route], self.routes[route])

        self.step_counter += 1
        df_info = self._get_info_vehicles()
        df_info['nr_step'] = deepcopy(self.step_counter)
        self.info_records.append(df_info)

        df_disp = self._display_info()

        return [], None, 0, df_disp
    
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
        df_info = self._get_info_vehicles()
        df_info['nr_step'] = deepcopy(self.step_counter)
        self.info_records.append(df_info)

        df_disp = self._display_info()
        
        return [], None, 0, df_disp
    
    def get_headways(self, info, control_vehs, terminal=False):
        control_veh = self.vehicles[control_vehs.index[0]]

        if len(control_veh.next_trips) == 0:
            print(control_vehs.iloc[0])
            print(control_veh.past_trips, control_veh.curr_trip, 
                  control_veh.next_trips)

        next_trip = control_veh.next_trips[0]
        schd_dep_t = next_trip.schedule['departure_time_sec'].values[0]

        earliest = max(schd_dep_t-MAX_EARLY_DEV*60, self.time)
        latest = max(schd_dep_t+MAX_LATE_DEV*60, self.time)

        if terminal:
            layover_bus_dep = get_layover_bus_dep(info, control_vehs)
            if layover_bus_dep is not None:
                earliest = max(earliest, layover_bus_dep)

        pre_hw, next_hw = control_veh.compute_headways(
            self.lines[control_veh.route_id], earliest, 
            self.routes[control_veh.route_id], terminal=terminal)
        return pre_hw, next_hw, earliest, latest

class FlexSimEnv(SimEnv):
    """"""

    def reset(self):
        super().__init__()
