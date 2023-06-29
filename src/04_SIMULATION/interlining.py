import pandas as pd
import numpy as np
from constants import *
from sim_env import recommended_dep_t
from copy import deepcopy



def check_for_interlining(rt_info, time, max_delay):
    missing_trip_exists = rt_info.loc[2, 'confirmed'] == 0
    within_otp_of_missing = rt_info.loc[2, 'ST'] + max_delay > time
    return missing_trip_exists and within_otp_of_missing

def can_lender_interline(rt_info):
    missing_trip_exists = rt_info.loc[2, 'confirmed'] == 0
    return not missing_trip_exists

def write_decision_vars(act_vars, counter_vars):
    decision_vars = {'var': [], 'actual': [], 'counter': []}
    decision_vars['var'].append('next departure')
    decision_vars['actual'].append(act_vars[0])
    decision_vars['counter'].append(counter_vars[0])

    decision_vars['var'].append('prior headway')
    decision_vars['actual'].append(act_vars[1])
    decision_vars['counter'].append(counter_vars[1])

    decision_vars['var'].append('headway')
    decision_vars['actual'].append(act_vars[2])
    decision_vars['counter'].append(counter_vars[2])

    decision_vars['var'].append('next headway')
    decision_vars['actual'].append(act_vars[3])
    decision_vars['counter'].append(counter_vars[3])
    return pd.DataFrame(decision_vars)

def compute_variables(df_):
    df = df_.copy()

    # this is for both actual and counter!
    prev_hw = df['AT'].iloc[1] - df['DT'].iloc[0]
    
    # before equalizing
    prev_at, prev_dt = df['AT'].iloc[1], df['DT'].iloc[1] 
    curr_rt = df['RT'].iloc[2]
    max_dep_t = df['DT_max'].iloc[2]
    next_rt = df['RT'].iloc[3]
    next_st = df['ST'].iloc[3]
    next_at = max(next_rt, next_st)

    # pre scenario
    hw = curr_rt - max(prev_at, prev_dt) # in case dt is nan
    next_hw = next_at - curr_rt

    # equalize
    new_at, new_hws, rec_wo_lim = recommended_dep_t(hw, next_hw, curr_rt, max_dep_t)
    variables = (new_at, prev_hw, *new_hws)
    return variables

def get_schd_hw(schd, rt, trip_seq):
     recent = schd[(schd['trip_sequence']<=trip_seq) & 
                   (schd['direction']==OUTBOUND_DIRECTIONS[rt]) &
                   (schd['stop_id']==OUTBOUND_TERMINALS[rt][0]) &
                   (schd['stop_sequence']==1)].copy()
     hws = recent['departure_sec'].diff()
     last_hw = hws.iloc[-1]
     return last_hw

def lend_variables(rt_info, veh_info, env, control_vehs):
    base = rt_info[rt_info['confirmed']==1].copy()
    base = base.reset_index(drop=True)
    base[['RT', 'DT_max']] = np.nan
    
    rt_id = base['route_id'].iloc[0]
    schd = env.routes[rt_id].schedule.copy()
    trip_seq = base['trip_sequence'].iloc[2]
    schd_hw = get_schd_hw(schd, rt_id, trip_seq)

    actual = base.copy()
    actual.loc[2,'RT'] = env.time

    veh_block = actual.loc[3, 'block_id']
    veh_df = veh_info[veh_info['block_id']==veh_block].iloc[[0]].copy()
    veh = env.vehicles[veh_df.index[0]]
    report_time = pred_terminal_report_time(
        veh_df, veh, actual.loc[3, 'ST'], 
        env.lines[veh.route_id], env.time) 
    actual.loc[3, 'RT'] = report_time

    sub_cols = ['route_id','trip_sequence', 'ST_str', 'confirmed',
                'ST', 'AT', 'DT', 'RT', 'trip_id', 'DT_max']
    actual_sub = actual[sub_cols].copy()
    trip_id = actual_sub['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    actual_sub.loc[2, 'DT_max'] = max_dep_t
    actual_vars = compute_variables(actual_sub)

    counter = base.copy()
    counter.loc[2, 'confirmed'] = 0
    counter = counter[counter['confirmed']==1].copy()
    counter = counter.reset_index(drop=True)
    
    for j in range(2, 4):
        veh_block = counter.loc[j, 'block_id']
        veh_df = veh_info[veh_info['block_id']==veh_block].iloc[[0]].copy()
        veh = env.vehicles[veh_df.index[0]]
        report_time = pred_terminal_report_time(
            veh_df, veh, counter.loc[j, 'ST'], 
            env.lines[veh.route_id], env.time) 
        counter.loc[j, 'RT'] = report_time

    counter_sub = counter[sub_cols].copy()
    trip_id = counter_sub['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    counter_sub.loc[2, 'DT_max'] = max_dep_t
    counter_vars = compute_variables(counter_sub)

    dec_vars = write_decision_vars(actual_vars, counter_vars)
    dec_vars['route_id'] = rt_id
    dec_vars['schd_hw'] = schd_hw

    veh = env.vehicles[control_vehs.index[0]]
    next_req_dep_t = None

    next_trip_ids = [t.id for t in veh.next_trips[:min(2, len(veh.next_trips))]] #TO-DO: revise?
    if len(veh.next_trips) > 2:
        t_schd = veh.next_trips[2].schedule.copy()
        next_req_dep_t = t_schd['departure_sec'].iloc[0]
    return actual_sub, counter_sub, dec_vars, next_req_dep_t, next_trip_ids

def borrow_variables(rt_info, veh_info, env): # veh_info is info[1] from main
    # remove cancelled trips after the current
    base = rt_info[~((rt_info.index>2) & 
                     (rt_info['confirmed']==0))].copy()
    base = base.reset_index(drop=True)
    base[['RT', 'DT_max']]= np.nan

    rt_id = base['route_id'].iloc[0]
    schd = env.routes[rt_id].schedule.copy()
    trip_seq = base['trip_sequence'].iloc[2]
    schd_hw = get_schd_hw(schd, rt_id, trip_seq)

    # without the trip
    actual = base[base['confirmed']==1].reset_index(drop=True)
    for j in range(2, 4):
        veh_block = actual.loc[j, 'block_id']
        veh_df = veh_info[veh_info['block_id']==veh_block].iloc[[0]].copy()
        veh = env.vehicles[veh_df.index[0]]
        report_time = pred_terminal_report_time(
            veh_df, veh, actual.loc[j, 'ST'], 
            env.lines[veh.route_id], env.time) 
        actual.loc[j, 'RT'] = report_time

    sub_cols = ['route_id','trip_sequence', 'ST_str', 'confirmed',
                'ST', 'AT', 'DT', 'RT', 'DT_max', 'trip_id']
    actual_sub = actual[sub_cols].copy()
    trip_id = actual_sub['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    actual_sub.loc[2, 'DT_max'] = max_dep_t
    actual_vars = compute_variables(actual_sub)

    # with the trip, index 2 is confirmed==0
    # but make sure index > 3 there are no confirmed == 0
    # and here we only need report time of index=3
    counter = base.copy()
    counter.loc[2, 'confirmed'] = 1
    counter.loc[2,'RT'] = max(env.time, 
                              counter.loc[2, 'ST']-MAX_EARLY_DEV*60) # this is from the current trip!
    counter = counter[counter['confirmed']==1].reset_index(drop=True)

    veh_block = counter.loc[3, 'block_id']
    veh_df = veh_info[veh_info['block_id']==veh_block].iloc[[0]].copy()
    veh = env.vehicles[veh_df.index[0]]
    report_time = pred_terminal_report_time(
        veh_df, veh, counter.loc[3, 'ST'], 
        env.lines[veh.route_id], env.time) 
    counter.loc[3, 'RT'] = report_time

    counter_sub = counter[sub_cols].copy()
    trip_id = counter_sub['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    counter_sub.loc[2, 'DT_max'] = max_dep_t
    counter_vars = compute_variables(counter_sub)

    dec_vars = write_decision_vars(actual_vars, counter_vars)
    dec_vars['route_id'] = rt_id
    dec_vars['schd_hw'] = schd_hw

    # at what time would it come back to terminal?
    ghost_block = counter.loc[2, 'block_id']
    ghost_trip = counter.loc[2, 'trip_id']
    ghost_arr_t, ghost_schds = get_new_arr_t(rt_id, ghost_block, ghost_trip, env)
    return actual_sub, counter_sub, dec_vars, ghost_arr_t, ghost_schds

def change_in_hw(dec_vars):
    hw = dec_vars[dec_vars['var']=='headway'].copy()
    act_hw, count_hw = hw['actual'].iloc[0], hw['counter'].iloc[0]
    schd_hw = hw['schd_hw'].iloc[0]
    act_ratio = act_hw/schd_hw
    counter_ratio = count_hw/schd_hw
    if counter_ratio < 0.8: # avoid making it smaller than scheduled headway
        return 0
    return abs(act_ratio - counter_ratio)

def pred_terminal_report_time(veh_df, veh, schd_dep, line, time):
    next_event_t = veh_df['next_event_t'].iloc[0].total_seconds()
    if veh_df['status'].iloc[0] == 0: # yet to report
        # the report time at initialization
        # may or may not be within the OTP so
        return max(next_event_t, schd_dep-MAX_EARLY_DEV*60)
    if veh_df['status'].iloc[0] in [1,4]: # reported (and adjusted)
        return next_event_t
    
    # dwell time status (2) at terminal is not considered
    # trips dwelling at terminal would have been flagged as
    # last trip in stop.report_last

    if veh_df['status'].iloc[0] in [2,3]:
        # this would only be opposite direction
        rt = veh_df['route_id'].iloc[0]
        direct = veh_df['direction'].iloc[0]
        assert direct == INBOUND_DIRECTIONS[rt]
        assert veh.stop_idx != 0
        # compute time until arrival at terminal
        trip = veh.curr_trip
        stop_idx_from = veh.stop_idx - 1
        stop_idx_to = len(trip.stops) - 1
        travel_time = line.time_between_two_stops(stop_idx_from, stop_idx_to, 
                                                  trip.stops, time)
        # travel_time = time_between_two_stops(line, stop_idx_from, stop_idx_to, 
        #                                           time, trip.stops, time)
        arrives = time + travel_time
        return max(schd_dep-MAX_EARLY_DEV*60, arrives)
    
def get_new_arr_t(rt_id, block_id, trip_id, env):
    schd = env.routes[rt_id].schedule.copy()
    deps = schd[(schd['block_id']==block_id) &
                (schd['stop_sequence']==1)].copy()
    next_out_trip_dep = deps.loc[
        deps['schd_trip_id']==trip_id, 'departure_sec'].iloc[0]
    next_in_trip_id = deps.loc[
        deps['departure_sec']>=next_out_trip_dep, 'schd_trip_id'].iloc[1]
    
    next_in_trip_schd = schd[schd['schd_trip_id']==next_in_trip_id].copy()
    next_in_trip_arr = next_in_trip_schd['departure_sec'].max()

    schds = schd[schd['schd_trip_id'].isin([trip_id, next_in_trip_id])].copy()
    return next_in_trip_arr, schds

def get_interlining_opp(info, control_vehs, rts_info, borrow_rts, env):
    lend_rt_id = control_vehs['route_id'].iloc[0]
    actual_dfs = []
    counter_dfs = []

    lst_dec_vars = []

    act, counter, dec_vars, next_req_dep_t, orig_trip_ids = lend_variables(rts_info[lend_rt_id], 
                                                                    info[1], env, control_vehs)

    impact = {lend_rt_id: change_in_hw(dec_vars)}

    actual_dfs.append(act)
    counter_dfs.append(counter)
    lst_dec_vars.append(dec_vars)

    best_borrow_rt = None
    best_bc_ratio = 1.2
    next_two_scheds = None
    next_dep_t = None

    for rt in borrow_rts:
        rt_info = rts_info[rt].copy()
        act, counter, dec_vars, ghost_arr_t, ghost_schds = borrow_variables(rt_info, info[1], env)
        impact[rt] = change_in_hw(dec_vars)

        actual_dfs.append(act)
        counter_dfs.append(counter)
        lst_dec_vars.append(dec_vars)

        if next_req_dep_t is None or next_req_dep_t > ghost_arr_t:
            benefit = change_in_hw(dec_vars)
            cost = impact[lend_rt_id]
            ratio = benefit/cost
            both_missing = ghost_schds['confirmed'].unique().shape[0] == 1
            if ratio > best_bc_ratio and both_missing:
                best_borrow_rt = deepcopy(rt)
                best_bc_ratio = ratio
                next_two_scheds = ghost_schds.copy()
                next_dep_t = dec_vars.loc[dec_vars['var']=='next departure', 'counter'].iloc[0]
    return best_borrow_rt, next_two_scheds, next_dep_t, orig_trip_ids, best_bc_ratio