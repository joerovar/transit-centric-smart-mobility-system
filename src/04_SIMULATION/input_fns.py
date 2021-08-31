import csv
import numpy as np
import pandas as pd
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def get_interval(t, len_i):
    interval = int(t/(len_i*60))
    return interval


def get_route(route_id, direction, path_trips, path_stop_times, start_time, end_time, nr_intervals, start_interval, interval_length):
    link_times = {}
    dates = ['2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06']
    all_trips = pd.read_csv(path_trips)
    route_trips = all_trips[all_trips['route_id'] == route_id]
    route_trips_east = route_trips[route_trips['direction'] == direction]

    trip_ids = route_trips_east['trip_id'].astype(str).tolist()
    route_stop_times = pd.read_csv(path_stop_times)
    route_peak_stop_times = pd.DataFrame(columns=route_stop_times.columns)
    for t in trip_ids:
        temp = route_stop_times[route_stop_times['trip_id'] == int(t[4:])].reset_index()
        if not temp.empty:
            temp = temp.sort_values(by='stop_sequence')
            dispatch_time = temp['arrival_time'][0]
            dispatch_time = datetime.strptime(dispatch_time, "%H:%M:%S")
            if start_time <= dispatch_time <= end_time:
                route_peak_stop_times = route_peak_stop_times.append(temp, ignore_index=True)
                for d in dates:
                    date_specific = temp[temp['event_time'].astype(str).str[:10] == d]
                    stop_id = date_specific['stop_id'].astype(str).tolist()
                    times = date_specific['avl_sec'].tolist()
                    check_sequence = date_specific['stop_sequence'].tolist()
                    for i in range(len(stop_id)-1):
                        if check_sequence[i] == check_sequence[i + 1] - 1:
                            link = stop_id[i]+'-'+stop_id[i+1]
                            exists = link in link_times
                            if not exists:
                                link_times[link] = [[] for i in range(nr_intervals)]
                            nr_bin = get_interval(times[i] % 86400, interval_length) - start_interval
                            if nr_bin < nr_intervals:
                                link_times[link][nr_bin].append(times[i+1] - times[i])

    mean_link_times = {}
    stdev_link_times = {}
    for link in link_times:
        mean_link_times[link] = []
        stdev_link_times[link] = []
        for b in link_times[link]:
            mean_link_times[link].append(np.array(b).mean())
            stdev_link_times[link].append(np.array(b).std())
    some_trip_id = route_peak_stop_times['trip_id'][0]
    some_trip = route_peak_stop_times[route_peak_stop_times['trip_id'] == some_trip_id]
    some_trip = some_trip.sort_values(by='stop_sequence')
    stops = some_trip['stop_id'].astype(str).tolist()
    stops = list(dict.fromkeys(stops))
    return stops, mean_link_times, stdev_link_times


def get_demand(path, stops, nr_intervals, interval_length, start_interval):
    arr_rates = {}
    drop_rates = {}
    alight_fractions = {}
    # first we turn it into an OD matrix
    od_pairs = pd.read_csv(path)
    viable_dest = {}
    viable_orig = {}
    # since stops are ordered, stop n is allowed to pair with stop n+1 until N
    for i in range(len(stops)):
        if i == 0:
            viable_dest[stops[i]] = stops[i+1:]
        elif i == len(stops) - 1:
            viable_orig[stops[i]] = stops[:i]
        else:
            viable_dest[stops[i]] = stops[i + 1:]
            viable_orig[stops[i]] = stops[:i]
    for s in stops:
        # record arrival rate
        arrivals = od_pairs[od_pairs['BOARDING_STOP'].astype(str).str[:-2] == s].reset_index()
        arrivals = arrivals[arrivals['INFERRED_ALIGHTING_GTFS_STOP'].astype(str).str[:-2].isin(viable_dest.get(s, []))].reset_index()
        # record drop-offs
        dropoffs = od_pairs[od_pairs['INFERRED_ALIGHTING_GTFS_STOP'].astype(str).str[:-2] == s].reset_index()
        dropoffs = dropoffs[dropoffs['BOARDING_STOP'].astype(str).str[:-2].isin(viable_orig.get(s, []))].reset_index()

        arr_pax = []
        drop_pax = []
        for i in range(start_interval, start_interval + nr_intervals):
            temp_arr_pax = sum(arrivals[arrivals['bin_5'] == i]['mean'].tolist())
            arr_pax.append(temp_arr_pax/interval_length)
            temp_drop_pax = sum(dropoffs[dropoffs['bin_5'] == i]['mean'].tolist())
            drop_pax.append(temp_drop_pax/interval_length)
        arr_rates[s] = arr_pax
        drop_rates[s] = drop_pax
    dep_vol = {}
    prev_vol = [0] * nr_intervals
    for i in range(len(stops)):
        alight_fractions[stops[i]] = []
        for pv, d in zip(prev_vol, drop_rates[stops[i]]):
            try:
                alight_fractions[stops[i]].append(d/pv)
            except ZeroDivisionError:
                alight_fractions[stops[i]].append(0.0)
        dep_vol[stops[i]] = [round(pv + a - d, 2) for pv, a, d in zip(prev_vol, arr_rates[stops[i]], drop_rates[stops[i]])]
        prev_vol = dep_vol.get(stops[i])
    return arr_rates, alight_fractions


def read_scheduled_departures(path):
    sch_dep = []
    with open(path) as f:
        rf = csv.reader(f, delimiter=' ')
        for row in rf:
            sch_dep.append(int(row[0]) / 1e+09)
    return sch_dep


def _get_vol_profile(ons_minus_offs):
    dep_vol = []
    prev_dep_vol = 0
    for o in ons_minus_offs:
        dep_vol.append(prev_dep_vol+o)
        prev_dep_vol = dep_vol[-1]
    return dep_vol


def _get_passenger_rates(od, stops):
    ar = {}
    af = {}
    ons = od.sum(axis=1)
    offs = od.sum(axis=0)

    delta = np.subtract(ons, offs)
    vol = _get_vol_profile(delta)
    o = offs[1:]
    v = vol[:-1]
    p_alight = np.divide(o, v, out=np.zeros_like(o), where=v != 0)
    p_alight = np.append([0], p_alight)
    for i, j in zip(stops, ons):
        ar[i] = j
    for i, j in zip(stops, p_alight):
        af[i] = j
    pv = max(vol)
    return ar, af, pv


def extract_network(path):
    with open(path, 'r') as f:
        rf = csv.reader(f, delimiter=' ')
        route_stops = {}
        link_times = {}
        i = 0
        for row in rf:
            if row[0] == 'route_id':
                i = 1
                continue
            if row[0] == 'o-d':
                i = 2
                continue
            if i == 1:
                route_stops[row[0]] = row[1:]
            if i == 2:
                link_times[row[0]] = float(row[1])
    return route_stops, link_times


def extract_demand(path, route_stops):
    arrival_rates = {}
    alighting_fractions = {}
    peak_volumes = {}

    with open(path, 'r') as f:
        rf = csv.reader(f, delimiter=' ')
        o_rows = []
        d_cols = []
        fixed_od = []
        all_od = {}
        i = 0
        for row in rf:
            if row[0] == 'fixed_od':
                i = 1
                fixed_od.append([])
                o_rows.append([])
                route_id = row[1]
                all_od[route_id] = []
                continue
            if row[0] == 'OD':
                all_od[route_id].append(row)
                d_cols.append(row[1:])
                continue
            if i == 1:
                all_od[route_id].append(row)
                o_rows[-1].append(row[0])
                fixed_od[-1].append(row[1:])

    for k in all_od:
        b = np.array(all_od[k])
        b = np.delete(b, 0, 0)
        b = np.delete(b, 0, 1)
        b = b.astype(float)
        new_rates, new_fractions, new_pk_vol = _get_passenger_rates(b, route_stops[k])
        arrival_rates.update(new_rates)
        alighting_fractions.update(new_fractions)
        peak_volumes[k] = new_pk_vol
        all_od[k] = b.tolist()
    return arrival_rates, alighting_fractions, peak_volumes


def plot_multiple_bar_charts(wta, wtc, lbls, pathname=False):
    w = 0.27
    bar1 = np.arange(len(wta.keys()))
    bar2 = [i + w for i in bar1]
    plt.bar(bar1, wta.values(), w, label=lbls[0], color='b')
    plt.bar(bar2, wtc.values(), w, label=lbls[1], color='r')
    plt.xticks(bar1, wta.keys(), rotation=90, fontsize=6)
    plt.tight_layout()
    plt.legend()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


def plot_bar_chart(var, pathname=False):
    plt.bar(var.keys(), var.values())
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
    return


