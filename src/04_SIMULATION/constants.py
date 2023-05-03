import os

# UNIVERSALLY USEFUL
SRC_PATH = os.path.dirname(os.path.realpath(__file__))
# the rest of the directory tree is assummed known
SIM_INPUTS_PATH = 'data/sim_in/'
SIM_OUTPUTS_PATH = 'data/sim_out/'

# SIMULATION DATA COLLECTION INPUTS
DATA_START_TIME = '05:00:00' # NOT EARLIER THAN 4AM
DATA_END_TIME = '21:00:00' # NOT LATER THAN 9PM
GTFS_ZIP_FILE = 'CTA_GTFS_2022_10.zip'
AVL_FILE = 'AVL_81_2022-10.csv'
ODX_FILE = 'ODX_81_2022-10.csv'
RUNS_FILE = 'Runs_81_2022-10.csv'
YEAR_MONTH = '2022-10'
START_DATE = '2022-10-03'
END_DATE = '2022-10-28'
HOLIDAYS = ['2022-10-10']
ROUTE = '81'
OUTBOUND_DIRECTION_STR = 'East'
INBOUND_DIRECTION_STR = 'West'
INTERVAL_LENGTH_MINS = 30


# SIMULATION FIXED INPUTS
START_TIME = '05:00:00'
END_TIME = '10:00:00'
CAPACITY = 60
DWELL_TIME_PARAMS = {
    'board': 2.1,
    'alight': 1.6,
    'acc_dec': 4.0,
    'error': 1.5
}
TERMINAL_REPORT_DELAYS = (-4.0, 2.0)
MAX_EARLY_DEV = 5.0
MAX_LATE_DEV = 5.0

# FOR RESULTS
KEY_STOP_NAMES = ['Jefferson Park', 'Cicero', 
                  'Pulaski', 'Kimball',
                  'Western']

ANIMATION = {
    'y_lim': (41.952583,41.984450),
    'x_lim': (-87.768183,-87.642206),
    'text_loc': (-87.769,41.953)
}
