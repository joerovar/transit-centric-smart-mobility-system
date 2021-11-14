from input import *


class Passenger:
    def __init__(self, orig_idx, dest_idx, arr_time):
        self.orig_idx = orig_idx
        self.dest_idx = dest_idx
        self.arr_time = arr_time
        self.board_time = 0.0
        self.alight_time = 0.0
        self.denied = 0
        self.wait_time = 0.0


class Stop:
    def __init__(self, stop_id):
        self.stop_id = stop_id
        self.last_bus_time = []
        self.prev_denied = 0
        self.passengers = []


stops = []
for i in range(len(STOPS)):
    stops.append(Stop(STOPS[i]))
    for j in range(i+1, len(STOPS)):
        od_rate = ODT[i,j] #ADJUST FOR TIMEEEEEE
        max_size = int(od_rate*(TOTAL_MIN/60)*2)
        temp_pax_interarr_times = np.random.exponential(3600/(od_rate), size=max_size)
        temp_pax_arr_times = np.cumsum(temp_pax_interarr_times)
        temp_pax_arr_times = temp_pax_arr_times[temp_pax_arr_times <= (TOTAL_MIN/60)*3600]
        temp_pax_arr_times += START_TIME_SEC
        for t in temp_pax_arr_times:
            stops[i].passengers.append(Passenger(i,j,t))