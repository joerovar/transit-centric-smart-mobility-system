import pandas as pd
import numpy as np
from constants import *
from copy import deepcopy

def recommended_dep_t(pre_hw, next_hw, t, max_dep_t):
    hw_diff = next_hw-pre_hw
    hold_wo_lim = max(0,hw_diff/2)
    rec_wo_lim = t + hold_wo_lim # without limit
    rec_dep_t = min(rec_wo_lim, max_dep_t)
    true_hold = rec_dep_t - t
    new_hws = pre_hw+true_hold, next_hw-true_hold
    return rec_dep_t, new_hws, rec_dep_t


def check_for_interlining(rt_info, time, max_delay):
    missing_trip_exists = rt_info.loc[2, 'confirmed'] == 0
    within_otp_of_missing = rt_info.loc[2, 'ST'] + max_delay > time
    return missing_trip_exists and within_otp_of_missing

def can_lender_interline(rt_info):
    missing_trip_exists = rt_info.loc[2, 'confirmed'] == 0
    return not missing_trip_exists

def write_decision_vars(act_vars, counter_vars):
    decision_vars = {'var': [], 'actual': [], 'counter': []}

    cols = ['next departure', 'prior headway',
            'headway', 'next headway']
    
    for i in range(len(cols)):
        decision_vars['var'].append(cols[i])
        decision_vars['actual'].append(act_vars[i])
        decision_vars['counter'].append(counter_vars[i])
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

def predict_with_trip(base_df, env, info_vehs, sub_cols, rt_id,
                      borrower=False):
    ## base df is the set of next trips
    scenario = base_df.copy()
    if borrower:
        scenario.loc[2, 'confirmed'] = 1
        scenario = scenario[scenario['confirmed']==1]
        scenario = scenario.reset_index(drop=True)

    ## report time is now if lender
    if borrower:
        max_time = scenario.loc[2, 'ST']-MAX_EARLY_DEV*60
        scenario.loc[2, 'RT'] = max(env.time, max_time)
    else:
        scenario.loc[2,'RT'] = env.time
        
    trip_id = scenario['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    scenario.loc[2, 'DT_max'] = max_dep_t ## ensure limits

    ## we just need to predict the next trip
    report_time = pred_terminal_report_time(
        scenario, 3, info_vehs, env) 
    scenario.loc[3, 'RT'] = report_time

    return scenario[sub_cols]

def predict_without_trip(base_df, env, info_vehs, sub_cols, rt_id,
                         lender=False):
    scenario = base_df.copy()

    ## this is exclusive to the lender case
    if lender:
        scenario.loc[2, 'confirmed'] = 0

    ## exclude all missing trips
    scenario = scenario[scenario['confirmed']==1].copy()
    scenario = scenario.reset_index(drop=True)

    ## just record departure limit, report time later
    trip_id = scenario['trip_id'].iloc[2]
    _, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
    scenario.loc[2, 'DT_max'] = max_dep_t
    
    for j in range(2, 4):
        report_time = pred_terminal_report_time(
            scenario, j, info_vehs, env) 
        scenario.loc[j, 'RT'] = report_time        

    return scenario[sub_cols]

def lend_variables(rt_info, veh_info, env, control_vehs):
    ## first, create baseline
    base = rt_info[rt_info['confirmed']==1].copy()
    base = base.reset_index(drop=True)
    base[['RT', 'DT_max']] = np.nan

    ## collect useful information
    rt_id = base['route_id'].iloc[0]
    schd = env.routes[rt_id].schedule.copy()
    trip_seq = base['trip_sequence'].iloc[2]
    schd_hw = get_schd_hw(schd, rt_id, trip_seq)

    sub_cols = ['route_id','trip_sequence', 
                'ST_str', 'confirmed', 'ST', 
                'AT', 'DT', 'RT', 'DT_max', 'trip_id', 'block_id']
   
    ## get next trips for actual scenario (with trip)
    actual = predict_with_trip(base, env, veh_info, 
                               sub_cols, rt_id)
    actual_vars = compute_variables(actual)

    ## get next trips for counterfactual scenario (without trip)
    counter = predict_without_trip(base, env, veh_info,
                                   sub_cols, rt_id, lender=True)
    counter_vars = compute_variables(counter)

    ## write it all up
    dec_vars = write_decision_vars(actual_vars, counter_vars)
    dec_vars['route_id'] = rt_id
    dec_vars['schd_hw'] = schd_hw

    scenarios = {'actual': actual,
                 'counterfactual': counter}

    ## 
    veh = env.vehicles[control_vehs.index[0]]

    ## find required return time to depart on-time
    next_req_dep_t = None
    next_trips = veh.next_trips[:min(2, len(veh.next_trips))]
    next_trip_ids = [t.id for t in next_trips] #TO-DO: revise?
    if len(veh.next_trips) > 2:
        t_schd = veh.next_trips[2].schedule.copy()
        next_req_dep_t = t_schd['departure_sec'].iloc[0]
    
    next_info = {'departure': next_req_dep_t,
                 'trip_ids': next_trip_ids}
    
    return scenarios, dec_vars, next_info

def borrow_variables(rt_info, info_vehs, env): 
    # info_vehs is info[1] from main

    ## first create baseline, remove later cancelled trips
    base = rt_info[~((rt_info.index>2) & 
                     (rt_info['confirmed']==0))].copy()
    base = base.reset_index(drop=True)
    base[['RT', 'DT_max']]= np.nan

    ## collect useful information
    rt_id = base['route_id'].iloc[0]
    schd = env.routes[rt_id].schedule.copy()
    trip_seq = base['trip_sequence'].iloc[2]
    schd_hw = get_schd_hw(schd, rt_id, trip_seq)

    sub_cols = ['route_id','trip_sequence', 
                'ST_str', 'confirmed', 'ST', 
                'AT', 'DT', 'RT', 'DT_max', 
                'trip_id', 'block_id']
    
    ## predict current scenario (without trip)

    actual = predict_without_trip(base, env, info_vehs, 
                                  sub_cols, rt_id)
    actual_vars = compute_variables(actual)

    ## counterfactual (with trip): 
    ## idx 2 cancelled, idx > 2 not
    # only report time of index=3
    counter = predict_with_trip(base, env, info_vehs, sub_cols,
                                rt_id, borrower=True)
    counter_vars = compute_variables(counter)

    ## write it all up
    dec_vars = write_decision_vars(actual_vars, counter_vars)
    dec_vars['route_id'] = rt_id
    dec_vars['schd_hw'] = schd_hw
    
    scenarios = {'actual': actual,
                 'counterfactual': counter}

    # at what time would it come back to terminal?
    ghost_block = counter.loc[2, 'block_id']
    ghost_trip = counter.loc[2, 'trip_id']
    ghost_arr_t, ghost_schds = get_next_schedules(
        rt_id, ghost_block, ghost_trip, env)
    
    next_info = {'return': ghost_arr_t,
                 'schedules': ghost_schds}
    return scenarios, dec_vars, next_info

def change_in_hw(dec_vars):
    hw = dec_vars[dec_vars['var']=='headway'].copy()
    act_hw, count_hw = hw['actual'].iloc[0], hw['counter'].iloc[0]
    schd_hw = hw['schd_hw'].iloc[0]
    act_ratio = act_hw/schd_hw
    counter_ratio = count_hw/schd_hw
    if counter_ratio < 0.8: # avoid making it smaller than scheduled headway
        return 0
    return abs(act_ratio - counter_ratio)

def pred_terminal_report_time(next_trips, idx, info_vehs, env):
    block_id = next_trips.loc[idx, 'block_id']

    ## this is based on the info from env.step output
    veh_info = info_vehs[
        info_vehs['block_id']==block_id].iloc[[0]].copy()
    
    ## this is the object
    veh = env.vehicles[veh_info.index[0]]

    ## other variables
    time = env.time
    line = env.lines[veh.route_id]
    schd_dep = next_trips.loc[idx, 'ST']

    next_event_t = veh_info['next_event_t'].iloc[0].total_seconds()
    veh_status = veh_info['status'].iloc[0]

    assert veh_status in [0,1,2,3,4,5]

    if veh_status in [0,5]: # yet to report
        # the report time at initialization
        # may or may not be within the OTP so
        return max(next_event_t, schd_dep-MAX_EARLY_DEV*60)
    
    if veh_status in [1,4]: # reported (and adjusted)
        return next_event_t
    
    # dwell time status (2) at terminal is not considered
    # trips dwelling at terminal would have been flagged as
    # last trip in stop.report_last

    if veh_status in [2,3]:
        # this would only be opposite direction
        rt = veh_info['route_id'].iloc[0]
        direct = veh_info['direction'].iloc[0]
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
    
def get_next_schedules(rt_id, block_id, schd_trip_id, env):
    schd = env.routes[rt_id].schedule.copy()

    ## to know when it wil arrive
    block_deps = schd[(schd['block_id']==block_id) & 
                      (schd['stop_sequence']==1)].copy()
    block_deps = block_deps.sort_values(
        by='departure_sec').reset_index(drop=True)
    
    ## should be sorted, so locate and take first 2 indices
    trip_idx = block_deps[
        block_deps['schd_trip_id']==schd_trip_id].index[0]
    next_2_trip_ids = block_deps.loc[
        trip_idx:trip_idx+1, 'schd_trip_id'].tolist()
    
    ## get schedules and return time
    next_2_schds = schd[
        schd['schd_trip_id'].isin(next_2_trip_ids)].copy()
    return_time = next_2_schds['departure_sec'].max()

    return return_time, next_2_schds

def eval_interlining(info, control_vehs, rts_info, borrow_rts, env,
                     debug=False):
    lend_rt_id = control_vehs['route_id'].iloc[0]

    scenarios_L, vars_L, next_info_L = lend_variables(rts_info[lend_rt_id], 
                                                    info[1], env, control_vehs)

    impact = {lend_rt_id: change_in_hw(vars_L)}

    ## initial conditions
    best_switch = {'ratio': 1.2, 
                   'lend_trip_ids': next_info_L['trip_ids'],
                   'route': None,
                   'departure': None,
                   'schedules': None,
                   'scenarios': {'lend': scenarios_L, 'borrow': None},
                   'variables': {'lend': vars_L, 'borrow': None}}

    for rt in borrow_rts:
        rt_info = rts_info[rt].copy()
        scenarios_B, vars_B, next_info_B = borrow_variables(rt_info, info[1], env)
        req_return = next_info_L['departure']
        est_return = next_info_B['return']

        if debug:
            print(lend_rt_id, rt)
            print(req_return, est_return)

        if req_return is None or req_return >= est_return:
            benefit = change_in_hw(vars_B)
            cost = impact[lend_rt_id]
            ratio = benefit/cost
            
            if debug:
                print(lend_rt_id, rt)
                print(req_return, est_return)
                print(benefit, cost)
                print(ratio)
            ## is ghost return also missing?
            schds = next_info_B['schedules']
            both_missing = schds['confirmed'].unique().shape[0] == 1

            if ratio > best_switch['ratio'] and both_missing:
                best_switch['route'] = deepcopy(rt)
                best_switch['ratio'] = deepcopy(ratio)
                best_switch['schedules'] = next_info_B['schedules'].copy()
                next_dep = vars_B.loc[
                    vars_B['var']=='next departure', 'counter'].iloc[0]
                best_switch['departure'] = next_dep
                best_switch['scenarios']['borrow'] = scenarios_B
                best_switch['variables']['borrow'] = vars_B
    return best_switch