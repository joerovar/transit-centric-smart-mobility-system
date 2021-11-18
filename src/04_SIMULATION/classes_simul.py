from input import *


class Trip:
    def __init__(self, trip_id):
        self.trip_id = trip_id
        self.pax = []


class Passenger:
    def __init__(self, orig_idx, dest_idx, arr_time):
        self.orig_idx = orig_idx
        self.dest_idx = dest_idx
        self.arr_time = arr_time
        self.board_time = 0.0
        self.alight_time = 0.0
        self.denied = 0
        self.wait_time = 0.0
        self.journey_time = 0.0
        self.trip_id = 0


class Stop:
    def __init__(self, stop_id):
        self.stop_id = stop_id
        self.pax = []


# samp_stops = ['386', '388', '390']
# link_time = [150, 98]
# odt = np.array([[[0, 3, 4],
#                  [0, 0, 4],
#                  [0, 0, 0]],
#                 [[0, 2, 5],
#                  [0, 0, 6],
#                  [0, 0, 0]],
#                 [[0, 7, 3],
#                  [0, 0, 5],
#                  [0, 0, 0]]])
# interval_length_mins = 30
# start_time_sec = 36000
# focus_end_time_sec = 36000 + 4500
# init_headway = 270
# start_interval = int(start_time_sec / (60 * interval_length_mins))
# end_interval = int(focus_end_time_sec / (60 * interval_length_mins))
# sched_dep = 36100
# pax_init_time = [0] + link_time
# pax_init_time = np.array(pax_init_time).cumsum()
# pax_init_time += sched_dep - init_headway
# stops = []
# for i in range(len(samp_stops)):
#     stops.append(Stop(samp_stops[i]))
#     for j in range(i + 1, len(samp_stops)):
#         for k in range(start_interval, end_interval+1):
#             start_edge_interval = k * interval_length_mins * 60
#             end_edge_interval = start_edge_interval + interval_length_mins * 60
#             od_rate = odt[k-start_interval, i, j]
#             max_size = int(od_rate * (interval_length_mins / 60) * 3)
#             temp_pax_interarr_times = np.random.exponential(3600 / od_rate, size=max_size)
#             temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
#             if k == start_interval:
#                 temp_pax_arr_times += pax_init_time[i]
#             else:
#                 temp_pax_arr_times += max(start_edge_interval, pax_init_time[i])
#             temp_pax_arr_times = temp_pax_arr_times[temp_pax_arr_times <= min(focus_end_time_sec, end_edge_interval)]
#             for t in temp_pax_arr_times:
#                 stops[i].passengers.append(Passenger(i, j, t))
