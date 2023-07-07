import zipfile
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import math

def get_ratio(od, apc, stop_ky, apc_ky, bin_ky='bin_30mins'):
    tmp_tots = od.groupby([bin_ky, stop_ky])['pax'].sum().reset_index()
    tmp_tots = tmp_tots.merge(apc[[stop_ky, apc_ky, bin_ky]], on=[bin_ky, stop_ky])
    tmp_tots['ratio'] = tmp_tots[apc_ky]/tmp_tots['pax']
    return tmp_tots

def bpf(od, target_ons, target_offs):
    sum_ons, sum_offs = np.nansum(target_ons), np.nansum(target_offs)
    balance_target_factor = sum_ons / sum_offs if sum_offs else 0
    balanced_target_offs = target_offs * balance_target_factor
    od_temp = od.copy()
    # print('before')
    # print(np.round(np.nansum(od)), np.round(np.nansum(target_ons)), np.round(np.nansum(od)))
    for _ in range(15):
        # balance rows
        actual_ons = np.nansum(od_temp, axis=1)
        factor_ons = np.divide(target_ons, actual_ons, out=np.zeros_like(target_ons), where=actual_ons != 0)
        od_temp = od_temp * factor_ons[:, np.newaxis]
        actual_ons = np.nansum(od_temp, axis=1)
        # balance columns
        actual_offs = np.nansum(od_temp, axis=0)
        factor_offs = np.divide(balanced_target_offs, actual_offs, out=np.zeros_like(target_offs),
                                where=actual_offs != 0)
        od_temp = od_temp * factor_offs
        actual_offs = np.nansum(od_temp, axis=0)
        # to check for tolerance we first assign 1.0 to totals of zero which cannot be changed by the method
        factor_ons[actual_ons == 0] = 1.0
        factor_offs[actual_offs == 0] = 1.0
    # print('after')
    # print(np.round(np.nansum(od_temp)), np.round(np.nansum(target_ons)), np.round(np.nansum(od)))
    scaled_od_set = np.array(od_temp)
    return scaled_od_set

def link_times_from_avl(interval_mins, schd, route_avl):
    df_avl = route_avl.copy()
    df_avl = df_avl.sort_values(by=['date','trip_id', 'stop_sequence'])
    df_avl['diff_stop_seq'] = df_avl['stop_sequence'].diff().shift(-1)
    df_avl['next_arrival_time'] = df_avl['event_time'].shift(-1)
    df_avl['next_stop_id'] = df_avl['stop_id'].shift(-1)
    df_avl = df_avl[df_avl['diff_stop_seq']==1]
    df_avl['next_stop_id'] = df_avl['next_stop_id'].astype(int)
    df_avl['link_time'] = (df_avl['next_arrival_time'] - df_avl['departure_time']).dt.total_seconds()/60
    df_avl = df_avl[df_avl['link_time']>0]
    interval_col_str = 'bin_'+str(interval_mins)+'mins'
    df_avl[interval_col_str] = (
        ((df_avl['departure_time']-df_avl['departure_time'].dt.floor('d')).dt.total_seconds())/(60*interval_mins)).astype(int)
    
    link_times = df_avl[[
        interval_col_str, 'date', 'route_id', 'direction', 'trip_id',
        'stop_id', 'next_stop_id', 'stop_sequence', 'link_time']].copy()
    
    schedule = schd.copy()
    schedule['arrival_time'] = pd.to_timedelta(schedule['arrival_time'])
    schedule = schedule.sort_values(by=['schd_trip_id', 'stop_sequence'])
    schedule['diff_stop_seq'] = schedule['stop_sequence'].diff().shift(-1)
    schedule['next_arrival_time'] = schedule['arrival_time'].shift(-1)
    schedule['next_stop_id'] = schedule['stop_id'].shift(-1)
    schedule_copy = schedule[~schedule['next_stop_id'].isna()].copy()
    schedule_copy['next_stop_id'] = schedule_copy['next_stop_id'].astype(int)
    schedule_copy = schedule_copy[schedule_copy['diff_stop_seq']==1]
    schedule_copy['schd_link_time'] = (schedule_copy['next_arrival_time'] - schedule_copy['arrival_time']).dt.total_seconds()/60
    schd_link_times = schedule_copy[['schd_trip_id','stop_id', 'next_stop_id','schd_link_time']].copy()
    schd_link_times = schd_link_times.rename(columns={'schd_trip_id':'trip_id'})

    link_times = link_times.merge(
        schd_link_times, on=['trip_id', 'stop_id', 'next_stop_id'])
    
    return link_times

def get_apc_idxs(df, stop_ids): ## from APC flows dataframe
    df['bin_idx'] = df['bin_30mins'] + 1
    stop_df = pd.DataFrame(stop_ids, columns=['stop_id'])
    stop_df['stop_idx'] = stop_df.index + 1 ## this is to match the bin index logic
    df = df.merge(stop_df, on='stop_id')
    idxs = df[['bin_idx', 'stop_idx']].to_numpy().transpose()-1
    return idxs, stop_df


def get_odf_zeros(df, stop_ids): ## from OD flows dataframe
    dates = df['date'].unique()
    bins = df['bin_30mins'].unique()

    ## aware that this can cause problems in the O-D where either
    ## O or D are terminals 
    od_zeros = pd.DataFrame(
        list(product(bins, stop_ids, stop_ids, dates)), 
        columns=['bin_30mins','boarding_stop', 'alighting_stop','date'])

    od_zeros['date'] = pd.to_datetime(od_zeros['date'])

    return od_zeros

def get_od_numpy(df, stop_ids, idxs):
    ## create numpy out of OD flows
    od = np.zeros(
        shape=(df['bin_30mins'].max()+1, 
            stop_ids.shape[0], stop_ids.shape[0]))

    od[idxs[0], idxs[1], idxs[2]] = df['pax'].tolist()
    return od

def get_apc_numpy(od_df, apc_df, stop_ids, idxs):
    ## create numpy of APCs
    ons = np.zeros(
        shape=(od_df['bin_30mins'].max(), stop_ids.shape[0]))
    offs = np.zeros(
        shape=(od_df['bin_30mins'].max(), stop_ids.shape[0]))

    ons[idxs[0], idxs[1]] = apc_df['ons'].tolist()
    offs[idxs[0], idxs[1]] = apc_df['offs'].tolist()

    return ons, offs

def get_od_idxs(df, stop_df):
    ## assign index according to stop and bin unique idx
    for ss in ('boarding_stop', 'alighting_stop'):
        df = df.merge(
            stop_df, left_on=ss, right_on='stop_id')
        df = df.rename(columns={'stop_idx': ss + '_idx'})
    df = df.drop(['stop_id_x', 'stop_id_y'], axis=1)
    df['bin_30mins'] = df['bin_30mins'].astype(int)

    df['bin_idx'] = df['bin_30mins'] + 1
    idxs = df[
        ['bin_idx', 'boarding_stop_idx', 
         'alighting_stop_idx']].to_numpy().transpose()-1
    return idxs

def get_od_and_idxs(df, stop_ids, stop_df):
    odf = df[(df['boarding_stop'].isin(stop_ids)) &
             (df['alighting_stop'].isin(stop_ids))].copy()
    
    ## OD flows with zeros so the mean is not distorted!
    odf_zeros = get_odf_zeros(odf, stop_ids)

    odf = odf_zeros.merge(
        odf, how='left', 
        on=['bin_30mins', 'boarding_stop', 
            'alighting_stop', 'date']).fillna(0)
    
    odf_avg = odf.groupby(
        by=['boarding_stop', 
            'alighting_stop', 
            'bin_30mins'])['pax'].mean().reset_index()

    odf_avg = odf_avg[odf_avg['pax'] > 0].reset_index(drop=True)
    od_idxs = get_od_idxs(odf_avg, stop_df)

    return odf_avg, od_idxs

def scale_od_npy(od, ons, offs, bins):
    ## create numpy of soon-to-be scaled OD
    od_scaled = np.zeros_like(od)
    od_dict = {
        'boarding_stop_idx': [],
        'alighting_stop_idx': [],
        'pax': [], 
        'bin_30mins': []
    }

    for i in range(bins[0], bins[1]):
        ## initialize with 0.01 wherever zero in the OD matrix
        init_od = np.tri(od.shape[1], od.shape[1], -1)
        init_od = init_od.transpose()*0.01 + od[i]
        od_scaled[i] = bpf(init_od, ons[i], offs[i])

        ## add elements to dictionary (later dataframe)
        idxs = np.nonzero(od_scaled[i])
        pax = od_scaled[i, idxs[0], idxs[1]]
        od_dict['boarding_stop_idx'] += list(idxs[0] + 1)
        od_dict['alighting_stop_idx'] += list(idxs[1] + 1)
        od_dict['pax'] += list(pax)
        od_dict['bin_30mins'] += [i]*pax.shape[0]
    od_df = pd.DataFrame(od_dict)
    return od_df



class GTFSHandler:
    def __init__(self, zf_path):
        self.zf = zipfile.ZipFile(zf_path)

        self.stops = pd.read_csv(self.zf.open('stops.txt'))
        self.stop_times = pd.read_csv(self.zf.open('stop_times.txt'), dtype={'trip_id': str})
        self.calendar = pd.read_csv(self.zf.open( 'calendar.txt'), dtype={'service_id': str})        
        self.calendar['start_date_dt'] = pd.to_datetime(self.calendar['start_date'], format='%Y%m%d')
        self.calendar['end_date_dt'] = pd.to_datetime(self.calendar['end_date'], format='%Y%m%d')
        self.trips = pd.read_csv(self.zf.open('trips.txt'), 
                                 dtype={'service_id': str, 'trip_id': str,
                                        'shape_id': str, 'schd_trip_id': str})
        self.route_trips = None
        self.route_stop_times = None
        self.route_stops = None
        self.schedule = None

    def load_network(self, start_date, end_date, routes, weekdays_only=True):
        df_cal = self.calendar.copy()

        if weekdays_only:
            df_cal = df_cal[
            (df_cal['start_date_dt']<=pd.to_datetime(start_date)) 
            & (df_cal['end_date_dt']>=pd.to_datetime(end_date))].copy()

            service_ids = df_cal.loc[
                (df_cal['monday']==1) |
                (df_cal['tuesday']==1) |
                (df_cal['wednesday']==1) |
                (df_cal['thursday']==1) |
                (df_cal['friday']==1), 'service_id'].tolist()
            
        df_trips = self.trips[
            (self.trips['route_id'].isin(routes)) & 
            (self.trips['service_id'].isin(service_ids))
        ].copy()

        df_trips  = df_trips.merge(df_cal[[
            'service_id', 'monday','tuesday', 
            'wednesday', 'thursday','friday']], on='service_id')
        self.route_trips = df_trips.copy()

        df_stop_times = self.stop_times.copy()
        df_stop_times = df_stop_times[
            df_stop_times['trip_id'].isin(df_trips['trip_id'])].copy()
        df_stop_times = df_stop_times.merge(df_trips[
            ['trip_id', 'schd_trip_id', 
             'direction', 'block_id', 'route_id', 'shape_id',
             'monday', 'tuesday', 'wednesday', 'thursday', 'friday']], on='trip_id')
        df_stop_times = df_stop_times.sort_values(by='stop_sequence')
        df_stop_times['block_id'] = df_stop_times['block_id'].astype(int)
        
        self.route_stop_times = df_stop_times.copy()

        route_stops = df_stop_times[
            ['route_id', 'direction', 
             'stop_id', 'stop_sequence', 'shape_id']].copy().drop_duplicates()
        self.route_stops = route_stops.merge(
            self.stops[['stop_id', 'stop_name', 
                        'stop_lat', 'stop_lon']], on='stop_id').drop_duplicates().reset_index(drop=True)
    

    def load_schedule(self):
        stop_times = self.route_stop_times.copy()
        start_time_td = pd.to_timedelta(DATA_START_TIME)
        end_time_td = pd.to_timedelta(DATA_END_TIME)
        stop_times['departure_time_td'] = pd.to_timedelta(stop_times['departure_time'])
        
        only_deps = stop_times[(stop_times['stop_sequence']==1) & 
                               (stop_times['departure_time_td']>=start_time_td) &
                               (stop_times['departure_time_td']<=end_time_td)].copy()
        only_deps = only_deps.sort_values(by='departure_time_td')

        lst_dep_dfs = []

        for r in ROUTES:
            deps_out = only_deps[(only_deps['route_id']==r) & 
                                 (only_deps['direction']==OUTBOUND_DIRECTIONS[r])].copy().reset_index(drop=True)
            deps_in = only_deps[(only_deps['route_id']==r) & 
                                (only_deps['direction']==INBOUND_DIRECTIONS[r])].copy().reset_index(drop=True)

            # first inbound departure to be of the block of the first outbound departure
            first_block_out = deps_out['block_id'].iloc[0]
            first_in_dep_t = deps_in[deps_in['block_id']==first_block_out]['departure_time_td'].iloc[0]

            deps_in = deps_in[deps_in['departure_time_td'] >= first_in_dep_t]
            
            deps_out['trip_sequence'] = deps_out.index + 1
            deps_in['trip_sequence'] = deps_in.index + 1

            lst_dep_dfs.append(deps_out)
            lst_dep_dfs.append(deps_in)

        departs_both = pd.concat(lst_dep_dfs, ignore_index=True)

        stop_times['schd_trip_id'] = stop_times['schd_trip_id'].astype(int)
        departs_both['schd_trip_id'] = departs_both['schd_trip_id'].astype(int)

        schedule = stop_times.merge(departs_both[['schd_trip_id', 'trip_sequence']], on='schd_trip_id')
        self.schedule = schedule.drop(columns=['departure_time_td'])
        


class AVLHandler():
    def __init__(self, file_path):
        # insert AVL of desired route
        self.avl = pd.read_csv(file_path, parse_dates=['event_time', 'departure_time'])
        self.avl['date'] = self.avl['event_time'].dt.date
        self.avl['route_id'] = self.avl['route_id'].astype(int).astype(str)
        self.link_times = None

    def clean(self, schedule, holidays=None):
        # trip stop sequence will be used as identified to merge the scheduled time
        schedule = schedule.copy()
        schedule['trip_stop_id'] = schedule['schd_trip_id'].astype(str) + schedule['stop_id'].astype(str)
        # first clean up AVL from NaN trip IDs
        df_avl = self.avl[(self.avl['trip_id'].notna()) & (self.avl['stop_id'].notna())].copy()
        # clean up from those in the wee hours (12-1AM) which make it difficult to match to schedule
        df_avl = df_avl[
            (df_avl['event_time'].dt.hour > 4) &
            (df_avl['event_time'].dt.hour < 23)].copy()
        if holidays:
            df_avl = df_avl[~pd.to_datetime(df_avl['date']).isin(pd.to_datetime(holidays))].copy()

        df_avl['trip_id'], df_avl['stop_id'] = df_avl['trip_id'].astype(int), df_avl['stop_id'].astype(int)
        df_avl['trip_stop_id'] = df_avl['trip_id'].astype(str) + df_avl['stop_id'].astype(str)

        df_avl = df_avl.merge(
            schedule[['trip_stop_id','arrival_time', 'direction']], 
            on='trip_stop_id').drop_duplicates().reset_index(drop=True)

        # remove duplicates for same trip,stop,date
        print(f'With duplicates {df_avl.shape}')
        df_avl = df_avl.drop_duplicates(subset=['trip_stop_id','date'],keep='first').reset_index(drop=True)
        print(f'Without duplicates {df_avl.shape}')
        df_avl['schd_time'] = df_avl['event_time'].dt.floor('D') + pd.to_timedelta(df_avl['arrival_time'])
        df_avl = df_avl.drop(columns=['arrival_time', 'trip_stop_id'])

        self.avl = df_avl.copy()

    def get_link_times(self, interval_mins, schedule, routes):
        df_avl = self.avl.copy()
        link_times_dfs = []
        for r in routes:
            rt_avl = df_avl[df_avl['route_id']==r].copy()
            link_times_dfs.append(link_times_from_avl(interval_mins, schedule, rt_avl))
        self.link_times = pd.concat(link_times_dfs, ignore_index=True)
    

class ODXHandler():
    def __init__(self, odx_file_path, apc_df):
        # insert AVL of desired route
        self.odx = pd.read_csv(odx_file_path, dtype={'avl_bus_route': str})
        self.odx = self.odx.rename(columns={'avl_bus_route':'route_id'})

        self.apc = apc_df.copy()
        
        self.scaled_od = None

    def get_apc_flows(self):
        apc = self.apc.copy()
        # assign interval to each timestamp
        bin_size_mins = 30
        interval_col_str = 'bin_' + str(bin_size_mins) +'mins'
        time_sec = (apc['event_time']-apc['event_time'].dt.floor('d')).dt.total_seconds()
        apc[interval_col_str] = (time_sec/60/bin_size_mins).astype(int)

        apc['ons'] = apc['ron'] + apc['fon']
        apc['offs'] = apc['roff'] + apc['foff']
        apc = apc.drop(columns=['ron', 'fon', 'roff', 'foff'])

        apc_sum = apc.groupby(
            ['route_id','stop_id', 
            'date', 'bin_30mins', 
            'direction'])[['ons', 'offs']].sum().reset_index()

        apc_flows = apc_sum.groupby(
            ['route_id', 'stop_id', 
            'bin_30mins', 
            'direction'])[['ons', 'offs']].mean().reset_index()
        return apc_flows

    ## get_scaled_od
    def get_od_flows_per_day(self):
        od = self.odx.copy()
        od = od[od['inferred_alighting_gtfs_stop']!='None']

        od['transaction_dtm'] = pd.to_datetime(od['transaction_dtm'])
        od['date'] = od['transaction_dtm'].dt.normalize()
        # annoying long key
        od = od.rename(
            columns={'inferred_alighting_gtfs_stop': 'alighting_stop'})
        od['alighting_stop'] = od['alighting_stop'].astype(int)
        od['boarding_stop'] = od['boarding_stop'].astype(int)
        # remove holiday/weekend data and data between 11PM-5AM 
        od = od[
            (od['date'].dt.weekday < 5) &
            (~od['date'].isin(pd.to_datetime(HOLIDAYS))) &
            (od['transaction_dtm'].dt.hour < 23) &
            (od['transaction_dtm'].dt.hour > 4)
        ].copy()

        # assign interval to each timestamp
        bin_size_mins = 30
        interval_col_str = 'bin_'+str(bin_size_mins)+'mins'
        time_sec = (od['transaction_dtm']-od['transaction_dtm'].dt.floor('d')).dt.total_seconds()
        od[interval_col_str] = (time_sec/60/bin_size_mins).astype(int)

        # get pax counts for every day and interval obtained in data (absences are zeroes)
        od_flows = od.groupby(['route_id','boarding_stop', 
                                'alighting_stop', 'date', 
                                'bin_30mins'])['dw_transaction_id'].count().reset_index()
        od_flows = od_flows.rename(columns={'dw_transaction_id':'pax'})

        # remove None stops
        od_flows = od_flows.replace(
            to_replace='None', value=np.nan).dropna(subset=['boarding_stop', 
                                                            'alighting_stop'])
        return od_flows
    
    def scale_od(self):
        od_flows = self.get_od_flows_per_day()
        apc_flows = self.get_apc_flows()

        lst_sc_ods = []

        for r in ROUTES:
            odf_r = od_flows[od_flows['route_id']==r].copy()
            apcf_r = apc_flows[apc_flows['route_id']==r].copy()
            for direct in apcf_r['direction'].unique():
                ## get APC dataframe and indices
                apcf = apcf_r[apcf_r['direction']==direct].copy()
                stop_ids = apcf['stop_id'].unique()
                apc_idxs, stop_df = get_apc_idxs(apcf, stop_ids)

                ## get OD dataframe and indices
                odf, od_idxs = get_od_and_idxs(odf_r, stop_ids, stop_df)

                ## get arrays
                od_npy = get_od_numpy(odf, stop_ids, od_idxs)
                ons_npy, offs_npy = get_apc_numpy(odf, apcf, 
                                                stop_ids, apc_idxs)
                bin1,bin2 = odf['bin_30mins'].min(), odf['bin_30mins'].max()
                scaled_od_df = scale_od_npy(od_npy, ons_npy, offs_npy, (bin1, bin2))

                ## TO-DO make this more efficient
                scaled_od_df = scaled_od_df.merge(
                    stop_df, left_on='boarding_stop_idx', 
                    right_on='stop_idx')
                scaled_od_df = scaled_od_df.rename(
                    columns={'stop_id': 'boarding_stop'})
                scaled_od_df = scaled_od_df.merge(
                    stop_df, left_on='alighting_stop_idx', 
                    right_on='stop_idx')
                scaled_od_df = scaled_od_df.rename(
                    columns={'stop_id': 'alighting_stop'})
                scaled_od_df = scaled_od_df.drop(
                    ['boarding_stop_idx', 'alighting_stop_idx', 
                    'stop_idx_x', 'stop_idx_y'], axis=1)
                scaled_od_df['direction'] = direct
                scaled_od_df['route_id'] = r
                lst_sc_ods.append(scaled_od_df)

        scaled_od = pd.concat(lst_sc_ods, ignore_index=True)
        self.scaled_od = scaled_od

def extract_sim_data():
    # define line characteristics
    gtfs = GTFSHandler('data/gtfs/' + GTFS_ZIP_FILE)
    gtfs.load_network(START_DATE, END_DATE, ROUTES)
    gtfs.load_schedule()

    schedule = gtfs.schedule.copy()
    stops = gtfs.route_stops.copy()

    # NO NEED TO SPECIFY DATES SINCE THOSE COME FROM SCHEDULED TRIPS
    avl = AVLHandler('data/avl/' + AVL_FILE)
    avl.clean(schedule, holidays=HOLIDAYS)
    avl.get_link_times(INTERVAL_LENGTH_MINS, schedule, ROUTES)

    # demand data
    odx = ODXHandler('data/odx/' + ODX_FILE, avl.avl)
    odx.scale_od()
    return gtfs, avl, odx


def write_sim_data(gtfs, avl, odx):
    schedule = gtfs.schedule.copy()
    stops = gtfs.route_stops.copy()

    # odx.scale_od()
    # apc = odx.avg_apc_counts.copy()
    # scaled_od = odx.scaled_od.copy()
    # day_boards = scaled_od.groupby(['bin_30mins'])['pax'].sum().reset_index()
    # day_apc = apc.groupby(['bin_30mins'])['ons'].sum().reset_index()
    # day_boards = day_boards.merge(day_apc, on='bin_30mins')

    # fig, ax = plt.subplots()
    # ax.bar(day_boards['bin_30mins']-0.2, day_boards['pax'], 0.4, label='od')
    # ax.bar(day_boards['bin_30mins']+0.2, day_boards['ons'], 0.4, label='apc')
    # ax.set_xlabel('30min interval')
    # ax.legend()
    # plt.savefig('data/sim_in/validate/od_vs_apc.png')

    # compute complimentary measures (correlations)
    # df_link_times = get_full_length_trips(avl.link_times, ['East', 'West'])
    # correl_link_t = link_time_correlation(df_link_times, ['East', 'West'])
    # correl_run_t = run_time_correlation(df_link_times, ['East', 'West'], schedule)
    # df_lt = pd.DataFrame(correl_link_t)
    # df_rt = pd.DataFrame(correl_run_t)

    runs_df = pd.read_csv('data/cta/' + RUNS_FILE)
    runs_df = runs_df.rename(columns={'tripno':'schd_trip_id'})
    runs_df = runs_df.drop_duplicates('schd_trip_id')
    schedule = schedule.merge(runs_df[['schd_trip_id', 'runid']], on='schd_trip_id')

    # network_info = pd.concat([df_lt, df_rt]).reset_index(drop=True)

    # write files
    sim_data_path = 'data/sim_in/'
    # network_info.to_csv(sim_data_path + 'network_info.csv', index=False)
    stops.to_csv(sim_data_path + 'stops.csv', index=False)
    schedule.to_csv(sim_data_path + 'schedule.csv', index=False)
    avl.link_times.to_csv(sim_data_path + 'link_times.csv', index=False)
    odx.scaled_od.to_csv(sim_data_path + 'od.csv', index=False)
    return


def get_full_length_trips(df, directions):
    # df must be of link times
    # ASSUMING SINGLE PATTERN
    dfs = []
    # remove negative link times
    for di in directions:
        counts = df[df['direction']==di].groupby(
            ['date', 'trip_id'])['link_time'].count().reset_index()
        counts = counts.rename(columns={'link_time':'count'})
        full_length = counts[counts['count']==counts['count'].max()].copy()
        dfs.append(df.merge(full_length, on=['date', 'trip_id']).drop(columns=['count']).copy())
    df2 = pd.concat(dfs).reset_index(drop=True)
    return df2


def get_parts(initial, final, split=3):
    step = math.ceil((final-initial)/split)
    tmp = initial
    parts = []
    while tmp < final:
        parts.append(range(tmp, min(tmp+step, final)))
        tmp += step
    return parts

def link_time_correlation(df, directions):
    # df must be link times for fully recorded trips
    df2 = df.sort_values(by=['bin_30mins', 'date', 'trip_id', 'stop_sequence']).copy()
    df2['diff_stop_seq'] = df2['stop_sequence'].diff().shift(-1)
    df2['next_link_time'] = df2['link_time'].shift(-1)
    correlations = {'direction':[],
                    'parameter': [],
                    'value': []}
    for di in range(len(directions)):
        di_df = df2[df2['direction']==directions[di]].copy()
        stop_ranges = get_parts(2, di_df['stop_sequence'].max()) # list of ranges without first/last links
        for r in stop_ranges:
            tmp_correlation = []
            for stop_seq in r:
                tmp_df = di_df[(di_df['stop_sequence']==stop_seq) & (di_df['diff_stop_seq']==1)].copy()
                tmp_correlation.append(tmp_df[
                    ['link_time', 'next_link_time']].corr(method='pearson').loc['link_time', 'next_link_time'])
            correlations['value'].append(np.mean(tmp_correlation))
            correlations['parameter'].append('pearson_link_times_stops_' + str(list(r)[0]) + '-' + str(list(r)[-2]))
            correlations['direction'].append(directions[di])
    return correlations

def run_time_correlation(df, directions, sched):
    # df must be link times for fully recorded trips
    run_times = df.groupby(['date', 'trip_id'])['link_time'].sum().reset_index()
    run_times = run_times.rename(columns={'link_time':'run_time'})
    run_times = run_times.merge(
        sched[['schd_trip_id', 'trip_sequence','direction']], 
        left_on='trip_id', right_on='schd_trip_id').drop_duplicates().drop(columns='schd_trip_id')
    run_times = run_times.sort_values(by=['date','trip_sequence'])
    run_times['diff_trip_seq'] = run_times['trip_sequence'].diff().shift(-1)
    run_times = run_times[run_times['diff_trip_seq']==1]
    run_times['next_run_time'] = run_times['run_time'].shift(-1)

    correlations = {'direction': [],
                    'parameter': [],
                    'value': []}
    for di in range(len(directions)):
        run_ts_dir = run_times[run_times['direction']==directions[di]]
        correlations['direction'].append(directions[di])
        correlations['parameter'].append('pearson_run_time')
        correlations['value'].append(
            run_ts_dir[['run_time', 'next_run_time']].corr(method='pearson').loc['run_time', 'next_run_time']) 
    return correlations

# def evaluate_predictions():
#     var = np.load('data/odx/route_81_onboard_VAR_30min.npy')

#     pred_ons = list(var.flatten())
#     col_stops = (list(range(1,49)) + list(range(1,51)))*1156
#     col_direction = (['East']*48 + ['West']*50)*1156
#     col_bin = []
#     for i in range(1156):
#         col_bin += [i]*98
#     len(col_direction), len(col_stops), len(pred_ons), len(col_bin)
#     df_prd_ons = pd.DataFrame(zip(col_direction, col_stops, col_bin,pred_ons), columns=['direction', 'stop_sequence', 'bin','pred_ons'])

#     avl = AVLHandler('data/avl/' + AVL_FILE)

#     # define line characteristics
#     gtfs = GTFSHandler('data/gtfs/' + GTFS_ZIP_FILE, YEAR_MONTH)
#     gtfs.load_network(START_DATE, 
#     END_DATE, ROUTE, OUTBOUND_DIRECTION_STR, INBOUND_DIRECTION_STR)
#     gtfs.load_schedule(DATA_START_TIME, DATA_END_TIME, ('East', 'West'))

#     schedule = gtfs.schedule
#     avl.clean(schedule, holidays=HOLIDAYS)

#     apc  = avl.avl.copy()
#     apc = apc[apc['event_time'] <= pd.to_datetime('2022-10-24 00:00')]
#     apc['bin'] = (
#                 ((apc['event_time']-pd.to_datetime('2022-10-01 00:00')).dt.total_seconds())/(60*30)).astype(int)
#     apc['ons'] = apc['ron'] + apc['fon']


#     apc_ons = apc.groupby(by=['direction', 'stop_sequence', 'bin'])['ons'].sum().reset_index()
#     apc_ons = apc_ons.drop(apc_ons[(apc_ons['direction']=='East') & (apc_ons['stop_sequence']==49)].index)
#     apc_ons = apc_ons.drop(apc_ons[(apc_ons['direction']=='West') & (apc_ons['stop_sequence']==51)].index)

#     ons_compare = apc_ons.merge(df_prd_ons, on=['direction', 'stop_sequence', 'bin'])
#     ons_compare.to_csv('onboarding_compare.csv', index=False)
#     return
