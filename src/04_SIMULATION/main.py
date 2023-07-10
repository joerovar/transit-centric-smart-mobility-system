from sim_env import FixedSimEnv
from sim_env import recommended_dep_t, flag_departure_event
from data_handlers import write_sim_data
import pandas as pd
import os
from constants import *
import numpy as np
from tqdm import tqdm
from interlining import eval_interlining
from objects import Trip

def ehd_message(tim, pre_hw, nxt_hw, schd_dep, 
                dtmax, dtmin, rec, act, exp_dep,
                rt_id):
    print(f'----')
    print(f'Time {td(tim)}')
    print(f'Route ID {rt_id}')
    print(f'Expected: {td(exp_dep)}')
    print(f'Headways: {td(pre_hw)} -- {td(nxt_hw)}')
    print(f'Schedule: {td(schd_dep)}')
    print(f'{td(dtmin)} < Departure < {td(dtmax)}')
    print(f'Recommended {td(rec)} ---> {td(act)}')
    return

def save_df(df, pth_from_cwd, filename):
    df.to_csv(
        os.path.join(SRC_PATH, 
                     pth_from_cwd, filename), index=False
    )
    return

def td(t):
    return pd.to_timedelta(round(t), unit='S')

def run_base(env, done, n_steps=0):
    pbar = tqdm(desc='while loop', total=MAX_STEPS)
    while not done and n_steps < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        n_steps += 1
        pbar.update(1)
    pbar.close()
    return env, n_steps

def run_ehd(env, done, n_steps=0, debug=False):
    pbar = tqdm(desc='while loop', total=MAX_STEPS)
    while not done and n_steps < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        n_steps += 1
        pbar.update(1)
        if done:
            continue

        ref_veh_df = flag_departure_event(info[1])

        if ref_veh_df.empty:
            continue

        ref_veh = env.vehicles[ref_veh_df.index[0]]
        rt_id, trip_id = ref_veh.route_id, ref_veh.next_trips[0].id
        min_dep_t, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
        # new min dep t refers to the potential imposed minimum
        # based on a layover bus that may be late
        pre_hw, new_min_dep_t = env.get_headway(info[1], ref_veh_df,
                                               min_dep_t, 
                                               terminal=True)
        if pre_hw is None:
            continue

        if pre_hw < 0:
            continue

        next_hw = env.get_next_headway(ref_veh_df, terminal=True,
                                       expected_dep=new_min_dep_t)       

        new_dep_t, new_hws, rec_wo_lim = recommended_dep_t(pre_hw, next_hw, new_min_dep_t, max_dep_t)
        # new_dep_t = min(max_dep_t, rec_dep_t)

        updated_info = env.adjust_departure(ref_veh_df, new_dep_t)

        if debug:
            next_trip = ref_veh.next_trips[0]
            schd_dep_t = next_trip.schedule['departure_sec'].values[0]
            ehd_message(env.time, pre_hw, next_hw,
                        schd_dep_t, max_dep_t, min_dep_t, 
                        rec_wo_lim, new_dep_t, new_min_dep_t, ref_veh.route_id)
            return env, info, n_steps, updated_info
    pbar.close()
    return env, n_steps

def run_di(env, done, n_steps=0, debug=False, debug_ehd=False):
    pbar = tqdm(desc='while loop', total=MAX_STEPS)
    while not done and n_steps < MAX_STEPS:
        next_obs, rew, done, info = env.step()
        n_steps += 1
        pbar.update(1)

        if done:
            continue

        ref_veh_df = flag_departure_event(info[1])

        if ref_veh_df.empty:
            continue

        ref_veh = env.vehicles[ref_veh_df.index[0]]

        rt_id, trip_id = ref_veh.route_id, ref_veh.next_trips[0].id
        min_dep_t, max_dep_t = env.terminal_dep_limits(rt_id, trip_id)
        # new min dep t refers to the potential imposed minimum
        # based on a layover bus that may be late
        pre_hw, new_min_dep_t = env.get_headway(info[1], ref_veh_df,
                                                min_dep_t, 
                                                terminal=True)
        if pre_hw is None:
            continue

        if pre_hw < 0:
            continue

        next_hw = env.get_next_headway(ref_veh_df, terminal=True,
                                       expected_dep=new_min_dep_t)       

        rts_info = env.get_rts_info()

        lend_rt_id = ref_veh.route_id
        borrow_rts = env.get_borrow_rts(rts_info, lend_rt_id) ## we include a check on lender

        if debug and '91' in borrow_rts:
            return env, info, n_steps, ref_veh_df, rts_info, borrow_rts

        if borrow_rts and rts_info[lend_rt_id] is not None:
            best_switch = eval_interlining(info, ref_veh_df, 
                                           rts_info, borrow_rts, env,
                                           debug=debug)
            # if debug:
            #     return env, info, n_steps, ref_veh_df, rts_info, best_switch
            if best_switch['route']:
                ## switch!!
                # mark in schedule and worklog the dropped trips
                for trip_id in best_switch['lend_trip_ids']:
                    env.routes[ref_veh.route_id].drop_trip(trip_id)

                # switch route
                ref_veh.route_id = best_switch['route']

                # change next two trips
                j = 0
                next_schds = best_switch['schedules']
                for trip_id in next_schds['trip_id'].unique():
                    # mark as confirmed in route schedule
                    env.routes[ref_veh.route_id].fill_trip(trip_id, ref_veh.block_id)

                    ## update in vehicle object
                    trip_sched = next_schds[next_schds['trip_id']==trip_id].copy()
                    trip_sched = trip_sched.sort_values(
                        by='departure_sec').reset_index(drop=True)
                    ref_veh.next_trips[j] = Trip(trip_sched)

                    j += 1

                # next event time and type
                upd_info = env.adjust_route_and_departure(
                    ref_veh_df, best_switch['departure'])
                
                ## log in route supervisor report
                date_dt = pd.to_datetime(env.hist_date)
                req_return = best_switch['req_return']
                req_return = date_dt + td(req_return) if req_return is not None else None
                report = [date_dt+ td(env.time), 
                          lend_rt_id, best_switch['route'],
                          best_switch['ratio'], 
                          date_dt + td(best_switch['est_return']),
                          req_return]
                env.superv.log.append(report)
                continue

        new_dep_t, new_hws, rec_wo_lim = recommended_dep_t(pre_hw, next_hw, 
                                                           new_min_dep_t, max_dep_t)

        updated_info = env.adjust_departure(ref_veh_df, new_dep_t)

        if debug_ehd:
            next_trip = ref_veh.next_trips[0]
            schd_dep_t = next_trip.schedule['departure_sec'].values[0]
            ehd_message(env.time, pre_hw, next_hw,
                        schd_dep_t, max_dep_t, min_dep_t, 
                        rec_wo_lim, new_dep_t, new_min_dep_t, ref_veh.route_id)
            return env, info, n_steps, updated_info
    pbar.close()
    return env, n_steps

def process_results(env, scenario):
    df_events = pd.concat(env.info_records, ignore_index=True) # for animations 
    df_events = df_events[df_events['active']==1]
    df_events = df_events[ANIMATION_COLS]
    df_events['scenario'] = scenario
    df_events['time'] = df_events['time'].round()
    df_pax = env.get_pax_records(scenario=scenario) # for pax experience
    df_trips = env.get_trip_records(scenario=scenario) # for trip experience
    return df_events, df_pax, df_trips

def experiments(n_days=4, dates=None, 
                scenarios=('NC', 'EHD', 'EHD-DI')):
    env = FixedSimEnv()
    if dates is None:
        unique_dates = env.link_times['date'].unique()
        dates = np.random.choice(unique_dates, replace=False, size=n_days)
    lst_pax = []
    lst_trips = []
    lst_events = []
    lst_worklogs = []
    lst_superlogs = []

    event_counter = 0 ## we don't want more than two episodes here

    for day in dates:
        # # METHOD 1
        # # np.random.seed(0)
        scenario = 'NC'
        if scenario in scenarios:
            next_obs, rew, done, info = env.reset(hist_date=day)
            print(env.hist_date, scenario)
            env, n_steps = run_base(env, done, n_steps=0)

            events, pax, trips = process_results(env, 'NC')
            lst_pax.append(pax)
            lst_trips.append(trips)
            if event_counter < 2:
                lst_events.append(events)
                event_counter += 1

        # METHOD 2 DON'T RESET DATE
        # np.random.seed(0)
        scenario = 'EHD'
        if scenario in scenarios:
            next_obs, rew, done, info = env.reset(hist_date=day)
            print(env.hist_date, scenario)
            env, n_steps = run_ehd(env, done, n_steps=0)

            events, pax, trips = process_results(env, 'EHD')
            lst_pax.append(pax)
            lst_trips.append(trips)
        ## no event

        # METHOD 3 INTERLINING!!!
        scenario = 'EHD-DI'
        if scenario in scenarios:
            next_obs, rew, done, info = env.reset(hist_date=day)
            print(env.hist_date, scenario)
            env, n_steps = run_di(env, done, n_steps=0)

            events, pax, trips = process_results(env, 'EHD-DI')
            lst_pax.append(pax)
            lst_trips.append(trips)
            if event_counter < 2:
                lst_events.append(events)
                event_counter += 1
            
            rt_wlogs = []
            for rt in ROUTES:
                rt_wlogs.append(env.routes[rt].worklog)
            wlog = pd.concat(rt_wlogs, ignore_index=True)
            wlog = wlog[(wlog['dropped'] == 1) | (wlog['filled']==1)]
            wlog['date'] = env.hist_date
            lst_worklogs.append(wlog)
            superlog = pd.DataFrame(env.superv.log, 
                                    columns=env.superv.cols)
            lst_superlogs.append(superlog)

    df_pax = pd.concat(lst_pax, ignore_index=True)
    df_trips = pd.concat(lst_trips, ignore_index=True)
    df_events = pd.concat(lst_events, ignore_index=True)
    df_wlogs = pd.concat(lst_worklogs, ignore_index=True)
    df_slogs = pd.concat(lst_superlogs, ignore_index=True)
    return df_pax, df_trips, df_events, df_wlogs, df_slogs

if __name__ == '__main__':
    # if not written yet
    # write_sim_data()
    
    tstamp = pd.Timestamp.today().strftime('%m%d-%H%M')
    path_from_cwd = 'data/sim_out/experiments_' + tstamp
    os.mkdir(os.path.join(SRC_PATH, path_from_cwd))

    pax, trips, events, wlogs = experiments()

    save_df(pax, path_from_cwd, 'pax.csv')
    save_df(trips, path_from_cwd, 'trips.csv')
    save_df(events, path_from_cwd, 'events.csv')
    save_df(wlogs, path_from_cwd, 'worklog.csv')
