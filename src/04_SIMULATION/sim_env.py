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
    
def flag_departure_event(info, direction):
    tmp_info = info[info['t_since_last'].notna()].copy()
    flag_vehicle = tmp_info[(tmp_info['direction']==direction) & 
                            (tmp_info['status'].isin([1,4])) & 
                            (tmp_info['t_since_last']==pd.to_timedelta(0, unit='S'))].copy()
    return flag_vehicle

def recommended_dep_t(pre_hw, next_hw, t):
    hw_diff = next_hw-pre_hw
    rec_dep_t = t + max(0,hw_diff/2)
    return rec_dep_t

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
        link_times = pd.read_csv(SIM_INPUTS_PATH + 'link_times.csv')
        schedule = pd.read_csv(SIM_INPUTS_PATH + 'schedule.csv')
        od = pd.read_csv(SIM_INPUTS_PATH + 'od.csv')

        for df in (stops, link_times, schedule, od):
            df['route_id'] = df['route_id'].astype(str)
        self.routes = {}
        self.lines = {}
        for route in ROUTES:
            rt_stops = stops[stops['route_id']==route].copy()
            rt_link_times = link_times[link_times['route_id']==route].copy()
            rt_schd = schedule[schedule['route_id']==route].copy()

            self.lines[route] = Line(
                route, rt_stops, rt_link_times, INTERVAL_LENGTH_MINS,
                OUTBOUND_DIRECTIONS[route], INBOUND_DIRECTIONS[route]) 
            self.routes[route] = FixedRoute(route, rt_schd, OUTBOUND_DIRECTIONS[route],
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
    
    def _display_latest(self):
        # display info 
        df_disp = self.info_records[-1].copy()
        df_disp = df_disp.sort_values(by='t_until_next')
        df_disp['time'] = pd.to_timedelta(df_disp['time'].round(), unit='S')
        df_disp['t_until_next'] = pd.to_timedelta(df_disp['t_until_next'].round(), unit='S')
        df_disp['next_event_t'] = pd.to_timedelta(df_disp['next_event_t'].round(), unit='S')
        df_disp['t_since_last'] = pd.to_timedelta(df_disp['t_since_last'].round(), unit='S')
        disp_cols = ['time', 'nr_step', 'id','active', 'status', 'status_desc', 
         'next_event', 'next_event_t', 't_until_next', 'stop_sequence', 
         'direction', 'pax_load', 't_since_last', 'route_id']
        return df_disp[disp_cols]


    def reset(self, reset_date=True):
        super().reset()
        self.info_records = []

        if reset_date:
            hist_date = np.random.choice(self.lines[ROUTES[0]].link_times['date'].unique())
            for rt in ROUTES:
                self.lines[rt].hist_date = hist_date
                confirmed_trips = self.lines[rt].link_times.loc[
                    self.lines[rt].link_times['date']==hist_date, 'trip_id'].tolist() # what was observed
                self.routes[rt].update_schedule(confirmed_trips=confirmed_trips) 

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
                                           rt)
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

        df_disp = self._display_latest()

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

        df_disp = self._display_latest()
        
        return [], None, 0, df_disp

class FlexSimEnv(SimEnv):
    """"""

    def reset(self):
        super().__init__()
