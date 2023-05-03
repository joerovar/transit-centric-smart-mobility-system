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
    balance_target_factor = np.nansum(target_ons) / np.nansum(target_offs)
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


class GTFSHandler:
    def __init__(self, zf_path, zf_folder):
        self.zf = zipfile.ZipFile(zf_path)

        self.stops = pd.read_csv(self.zf.open(zf_folder + '/stops.txt'))
        self.stop_times = pd.read_csv(self.zf.open(zf_folder + '/stop_times.txt'))
        self.calendar = pd.read_csv(self.zf.open(zf_folder + '/calendar.txt'))        
        self.calendar['start_date_dt'] = pd.to_datetime(self.calendar['start_date'], format='%Y%m%d')
        self.calendar['end_date_dt'] = pd.to_datetime(self.calendar['end_date'], format='%Y%m%d')
        self.trips = pd.read_csv(self.zf.open(zf_folder + '/trips.txt'))

        self.route_trips = None
        self.route_stop_times = None
        self.route_stops = None
        self.route_stops_outbound = None
        self.route_stops_inbound = None
        self.schedule = None

    def load_network(self, start_date, end_date, route, 
    outbound_direction, inbound_direction, weekdays_only=True):
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
                (df_cal['friday']==1), 'service_id'].astype(str).tolist()
            
        df_trips = self.trips[
            (self.trips['route_id']==route) & 
            (self.trips['service_id'].astype(str).isin(service_ids))
        ].copy()
        self.route_trips = df_trips.copy()

        df_stop_times = self.stop_times.copy()
        df_stop_times['trip_id'] = df_stop_times['trip_id'].astype(str)
        df_stop_times = df_stop_times[df_stop_times['trip_id'].isin(df_trips['trip_id'])].copy()
        df_stop_times = df_stop_times.merge(df_trips[
            ['trip_id', 'schd_trip_id', 'direction', 'block_id']], on='trip_id')
        df_stop_times = df_stop_times.sort_values(by='stop_sequence')
        df_stop_times['block_id'] = df_stop_times['block_id'].astype(int)
        
        self.route_stop_times = df_stop_times.copy()

        self.route_stops_outbound = df_stop_times.loc[
            df_stop_times['direction']==outbound_direction, ['stop_id', 'stop_sequence']].copy()
        self.route_stops_outbound = self.route_stops_outbound.merge(
            self.stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id').drop_duplicates().reset_index(drop=True)
        self.route_stops_outbound = self.route_stops_outbound.sort_values(by='stop_sequence').reset_index(drop=True)

        self.route_stops_inbound = df_stop_times.loc[
            df_stop_times['direction']==inbound_direction, ['stop_id', 'stop_sequence']].copy()
        self.route_stops_inbound = self.route_stops_inbound.merge(
            self.stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id').drop_duplicates().reset_index(drop=True)
        self.route_stops_inbound = self.route_stops_inbound.sort_values(by='stop_sequence').reset_index(drop=True)
    

    def load_schedule(self, data_start_time, data_end_time,
                      directions):
        stop_times = self.route_stop_times.copy()
        start_time_td = pd.to_timedelta(data_start_time)
        end_time_td = pd.to_timedelta(data_end_time)
        stop_times['departure_time_td'] = pd.to_timedelta(stop_times['departure_time'])
        
        departure_dfs = []

        for i in range(len(directions)):
            departure_dfs.append(
                stop_times[
                (stop_times['stop_sequence']==1) & 
                (stop_times['direction']==directions[i]) &
                (stop_times['departure_time_td']>=start_time_td) &
                (stop_times['departure_time_td']<=end_time_td)
                ].sort_values(by='departure_time_td').reset_index(drop=True).copy()
            )
        
            # WORKS ONLY FOR SINGLE PATTERNS
            departure_dfs[i]['trip_sequence'] = departure_dfs[i].index + 1

        first_depart_outbound = departure_dfs[0]['block_id'].iloc[0]
        first_depart_inbound = departure_dfs[1][
            departure_dfs[1]['block_id']==first_depart_outbound]
        departure_dfs[1] = departure_dfs[1][
            departure_dfs[1]['departure_time_td']>=first_depart_inbound['departure_time_td'].iloc[0]
        ]

        departs_both = pd.concat(departure_dfs)
        stop_times['schd_trip_id'] = stop_times['schd_trip_id'].astype(int)
        departs_both['schd_trip_id'] = departs_both['schd_trip_id'].astype(int)

        schedule = stop_times.merge(departs_both[['schd_trip_id', 'trip_sequence']], on='schd_trip_id')
        self.schedule = schedule.drop(columns=['departure_time_td'])
        

class AVLHandler():
    def __init__(self, file_path):
        # insert AVL of desired route
        self.avl = pd.read_csv(file_path)
        self.avl['event_time'] = pd.to_datetime(self.avl['event_time'])
        self.avl['departure_time'] = pd.to_datetime(self.avl['departure_time'])
        self.avl['date'] = self.avl['event_time'].dt.date
        self.link_times = None

    def clean(self, schedule, holidays=None):
        # trip stop sequence will be used as identified to merge the scheduled time
        schedule = schedule.copy()
        schedule['trip_stop_seq'] = schedule['schd_trip_id'].astype(str) + schedule['stop_sequence'].astype(str)
        # first clean up AVL from NaN trip IDs
        df_avl = self.avl[(self.avl['trip_id'].notna()) & (self.avl['stop_id'].notna())].copy()
        # clean up from those in the wee hours (12-1AM) which make it difficult to match to schedule
        df_avl = df_avl[
            (df_avl['event_time'].dt.hour > 4) &
            (df_avl['event_time'].dt.hour < 23)].copy()
        if holidays:
            df_avl = df_avl[~pd.to_datetime(df_avl['date']).isin(pd.to_datetime(holidays))].copy()

        df_avl['trip_id'], df_avl['stop_id'] = df_avl['trip_id'].astype(int), df_avl['stop_id'].astype(int)
        df_avl['trip_stop_seq'] = df_avl['trip_id'].astype(str) + df_avl['stop_sequence'].astype(str)

        df_avl = df_avl.merge(
            schedule[['trip_stop_seq','arrival_time', 'direction']], 
            on='trip_stop_seq').drop_duplicates().reset_index(drop=True)

        # remove duplicates for same trip,stop,date
        print(f'With duplicates {df_avl.shape}')
        df_avl = df_avl.drop_duplicates(subset=['trip_stop_seq','date'],keep='first').reset_index(drop=True)
        print(f'Without duplicates {df_avl.shape}')
        df_avl['schd_time'] = df_avl['event_time'].dt.floor('D') + pd.to_timedelta(df_avl['arrival_time'])
        df_avl = df_avl.drop(columns=['arrival_time', 'trip_stop_seq'])

        self.avl = df_avl.copy()

    def get_link_times(self, interval_mins, schedule):
        df_avl = self.avl.copy()
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
        
        self.link_times = df_avl[[
            interval_col_str, 'date', 'direction', 'trip_id',
            'stop_id', 'next_stop_id', 'stop_sequence', 'link_time']].copy()
        
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

        self.link_times = self.link_times.merge(
            schd_link_times, on=['trip_id', 'stop_id', 'next_stop_id'])

class ODXHandler():
    def __init__(self, odx_file_path, apc_df, stops_ob, stops_ib):
        # insert AVL of desired route
        self.odx = pd.read_csv(odx_file_path)
        self.apc = apc_df
        self.avg_apc_counts = None
        self.avg_hist_od = None
        self.scaled_od = None
        self.stops_outbound = stops_ob
        self.stops_inbound = stops_ib
    
    def get_apc(self):
        apc = self.apc.copy()
        # assign interval to each timestamp
        interval_col_str = 'bin_' + str(30) +'mins'
        apc[interval_col_str] = (
            ((apc['event_time']-apc['event_time'].dt.floor('d')).dt.total_seconds())/(60*30)).astype(int)

        apc_counts = apc.groupby(
            ['stop_id', 'date', 'bin_30mins'])[['ron', 'fon', 'roff', 'foff']].sum().reset_index()

        apc_counts['ons'] = apc_counts['ron'] + apc_counts['fon']
        apc_counts['offs'] = apc_counts['roff'] + apc_counts['foff']
        apc_counts = apc_counts.drop(columns=['ron', 'fon', 'roff', 'foff'])

        dates = apc_counts['date'].unique()
        stops = apc_counts['stop_id'].unique()
        bins = apc_counts['bin_30mins'].unique()
        apc_count_w_zeros = pd.DataFrame(list(product(bins, stops, dates)), 
                                        columns=['bin_30mins','stop_id', 'date'])

        apc_count_w_zeros = apc_count_w_zeros.merge(
            apc_counts, how='left', on=['bin_30mins', 'stop_id', 'date']).fillna(0)

        avg_apc_counts = apc_count_w_zeros.groupby(
            ['stop_id', 'bin_30mins'])[['ons', 'offs']].mean().reset_index()
        avg_apc_counts['boarding_stop'] = avg_apc_counts['stop_id'].copy()
        avg_apc_counts['alighting_stop'] = avg_apc_counts['stop_id'].copy()
        avg_apc_counts = avg_apc_counts.drop(columns=['stop_id'])

        # # balance ons and offs in APC
        # ratio = avg_apc_counts['ons'].sum()/avg_apc_counts['offs'].sum()
        # avg_apc_counts['offs'] = avg_apc_counts['offs'] * ratio

        self.avg_apc_counts = avg_apc_counts.copy()

    
    def scaled_flows_by_dir(self, direction, all_pax_count):
        stops = self.stops_outbound.copy() if direction == 'outbound' else self.stops_inbound.copy()
        on_stops = stops.loc[stops['stop_sequence']<stops.shape[0], 'stop_id']
        off_stops = stops.loc[stops['stop_sequence']>1, 'stop_id']
        apc = self.avg_apc_counts.copy()

        apc = apc[apc['boarding_stop'].isin(stops['stop_id'])]
        apc.loc[apc['boarding_stop']==14102, 'offs'] = 0.0
        apc.loc[apc['boarding_stop']==3773, 'ons'] = 0.0
        apc = apc.merge(stops[['stop_id', 'stop_sequence']], left_on='boarding_stop', right_on='stop_id')
        apc['bin_idx'] = apc['bin_30mins'] + 1

        xy = apc[['bin_idx', 'stop_sequence']].to_numpy().transpose()-1

        pax_count = all_pax_count[
            (all_pax_count['boarding_stop'].isin(on_stops)) &
            (all_pax_count['alighting_stop'].isin(off_stops))
        ].copy()

        dates = pax_count['date'].unique()
        bins = pax_count['bin_30mins'].unique()

        pax_count_w_zeros = pd.DataFrame(
            list(product(bins, on_stops, off_stops, dates)), 
            columns=['bin_30mins','boarding_stop', 'alighting_stop','date'])

        pax_count_w_zeros = pax_count_w_zeros.merge(
            pax_count, how='left', 
            on=['bin_30mins', 'boarding_stop', 'alighting_stop', 'date']).fillna(0)

        # # get avg pax counts for every interval (this is the non-scaled OD)
        avg_pax_count = pax_count_w_zeros.groupby(
            by=['boarding_stop', 
                'alighting_stop', 
                'bin_30mins'])['pax'].mean().reset_index()

        avg_pax_count = avg_pax_count[avg_pax_count['pax']>0].reset_index(drop=True)

        for ss in ('boarding_stop', 'alighting_stop'):
            avg_pax_count = avg_pax_count.merge(
                stops[['stop_id', 'stop_sequence']], left_on=ss, right_on='stop_id')
            avg_pax_count = avg_pax_count.rename(columns={'stop_sequence': ss + '_seq'})

        avg_pax_count = avg_pax_count.drop(['stop_id_x', 'stop_id_y'], axis=1)
        od = np.zeros(
            shape=(avg_pax_count['bin_30mins'].max()+1, stops.shape[0], stops.shape[0]))
        avg_pax_count['bin_idx'] = avg_pax_count['bin_30mins'] + 1
        xyz = avg_pax_count[
            ['bin_idx', 'boarding_stop_seq', 'alighting_stop_seq']].to_numpy().transpose()-1
        od[xyz[0], xyz[1], xyz[2]] = avg_pax_count['pax'].tolist()

        ons = np.zeros(shape=(avg_pax_count['bin_30mins'].max(), stops.shape[0]))
        offs = np.zeros(shape=(avg_pax_count['bin_30mins'].max(), stops.shape[0]))

        ons[xy[0], xy[1]] = apc['ons'].tolist()
        offs[xy[0], xy[1]] = apc['offs'].tolist()


        bin1,bin2 = avg_pax_count['bin_30mins'].min(), avg_pax_count['bin_30mins'].max()
        od_scaled = np.zeros_like(od)
        od_dict = {
            'boarding_stop_seq': [],
            'alighting_stop_seq': [],
            'pax': [], 
            'bin_30mins': []
        }

        for i in range(bin1, bin2):
            boost_od = np.tri(od.shape[1], od.shape[1], -1).transpose()*0.01 + od[i]
            od_scaled[i] = bpf(boost_od, ons[i], offs[i])
            idxs = np.nonzero(od_scaled[i])
            pax = od_scaled[i, idxs[0], idxs[1]]
            od_dict['boarding_stop_seq'] += list(idxs[0] + 1)
            od_dict['alighting_stop_seq'] += list(idxs[1] + 1)
            od_dict['pax'] += list(pax)
            od_dict['bin_30mins'] += [i]*pax.shape[0]

        od_df = pd.DataFrame(od_dict)
        od_df = od_df.merge(
            stops[['stop_sequence', 'stop_id']], 
            left_on='boarding_stop_seq', right_on='stop_sequence')
        od_df = od_df.rename(columns={'stop_id': 'boarding_stop'})
        od_df = od_df.merge(
            stops[['stop_sequence', 'stop_id']], 
            left_on='alighting_stop_seq', right_on='stop_sequence')
        od_df = od_df.rename(columns={'stop_id': 'alighting_stop'})
        od_df = od_df.drop(
            ['boarding_stop_seq', 'alighting_stop_seq', 
             'stop_sequence_x', 'stop_sequence_y'], axis=1)

        return od_df

    def get_scaled_od(self):
        df = self.odx.copy()
        df = df[df['inferred_alighting_gtfs_stop']!='None']

        df['transaction_dtm'] = pd.to_datetime(df['transaction_dtm'])
        df['date'] = df['transaction_dtm'].dt.normalize()
        # annoying long key
        df = df.rename(columns={'inferred_alighting_gtfs_stop': 'alighting_stop'})
        df['alighting_stop'] = df['alighting_stop'].astype(int)
        df['boarding_stop'] = df['boarding_stop'].astype(int)
        # remove holiday/weekend data and data between 11PM-5AM 
        df = df[
            (df['date'].dt.weekday < 5) &
            (~df['date'].isin(pd.to_datetime(HOLIDAYS))) &
            (df['transaction_dtm'].dt.hour < 23) &
            (df['transaction_dtm'].dt.hour > 4)
        ].copy()

        # assign interval to each timestamp
        interval_col_str = 'bin_'+str(30)+'mins'
        df[interval_col_str] = (
            ((df['transaction_dtm']-df['transaction_dtm'].dt.floor('d')).dt.total_seconds())/(60*30)).astype(int)

        # get pax counts for every day and interval obtained in data (absences are zeroes)
        pax_count = df.groupby(['boarding_stop', 
                                'alighting_stop', 'date', 
                                'bin_30mins'])['dw_transaction_id'].count().reset_index()
        pax_count = pax_count.rename(columns={'dw_transaction_id':'pax'})


        # remove None stops
        pax_count = pax_count.replace(
            to_replace='None', value=np.nan).dropna(subset=['boarding_stop', 
                                                            'alighting_stop'])
        
        scaled_od_o = self.scaled_flows_by_dir('outbound', pax_count)
        scaled_od_i = self.scaled_flows_by_dir('inbound', pax_count)

        self.scaled_od = pd.concat(
            [scaled_od_o, scaled_od_i]
        ).reset_index(drop=True)


def write_sim_data():
    # define line characteristics
    gtfs_handler = GTFSHandler('data/gtfs/' + GTFS_ZIP_FILE, YEAR_MONTH)
    gtfs_handler.load_network(START_DATE, 
    END_DATE, ROUTE, OUTBOUND_DIRECTION_STR, INBOUND_DIRECTION_STR)
    gtfs_handler.load_schedule(DATA_START_TIME, DATA_END_TIME, ('East', 'West'))

    schedule = gtfs_handler.schedule
    stops_outbound = gtfs_handler.route_stops_outbound
    stops_inbound = gtfs_handler.route_stops_inbound

    # NO NEED TO SPECIFY DATES SINCE THOSE COME FROM SCHEDULED TRIPS
    avl_handler = AVLHandler('data/avl/' + AVL_FILE)
    avl_handler.clean(schedule, holidays=HOLIDAYS)
    avl_handler.get_link_times(INTERVAL_LENGTH_MINS, schedule)

    # demand data
    odx_handler = ODXHandler('data/odx/' + ODX_FILE, 
                             avl_handler.avl, stops_outbound, 
                             stops_inbound)
    odx_handler.get_apc()
    odx_handler.get_scaled_od()
    # odx_handler.scale_od()
    apc = odx_handler.avg_apc_counts.copy()
    scaled_od = odx_handler.scaled_od.copy()
    day_boards = scaled_od.groupby(['bin_30mins'])['pax'].sum().reset_index()
    day_apc = apc.groupby(['bin_30mins'])['ons'].sum().reset_index()
    day_boards = day_boards.merge(day_apc, on='bin_30mins')

    fig, ax = plt.subplots()
    ax.bar(day_boards['bin_30mins']-0.2, day_boards['pax'], 0.4, label='od')
    ax.bar(day_boards['bin_30mins']+0.2, day_boards['ons'], 0.4, label='apc')
    ax.set_xlabel('30min interval')
    ax.legend()
    plt.savefig('data/sim_in/validate/od_vs_apc.png')

    # compute complimentary measures (correlations)
    df_link_times = get_full_length_trips(avl_handler.link_times, ['East', 'West'])
    correl_link_t = link_time_correlation(df_link_times, ['East', 'West'])
    correl_run_t = run_time_correlation(df_link_times, ['East', 'West'], schedule)
    df_lt = pd.DataFrame(correl_link_t)
    df_rt = pd.DataFrame(correl_run_t)

    runs_df = pd.read_csv('data/cta/' + RUNS_FILE)
    runs_df = runs_df.rename(columns={'tripno':'schd_trip_id'})
    runs_df = runs_df.drop_duplicates('schd_trip_id')
    schedule = schedule.merge(runs_df[['schd_trip_id', 'runid']], on='schd_trip_id')

    # write files
    sim_data_path = 'data/sim_in/'
    network_info = pd.concat([df_lt, df_rt]).reset_index(drop=True)
    network_info.to_csv(sim_data_path + 'network_info.csv', index=False)
    stops_outbound.to_csv(sim_data_path + 'stops_outbound.csv', index=False)
    stops_inbound.to_csv(sim_data_path + 'stops_inbound.csv', index=False)
    schedule.to_csv(sim_data_path + 'schedule.csv', index=False)
    avl_handler.link_times.to_csv(sim_data_path + 'link_times.csv', index=False)
    scaled_od.to_csv(sim_data_path + 'od.csv', index=False)
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
    # print(dfs[0].shape[0], dfs[1].shape[0])
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
