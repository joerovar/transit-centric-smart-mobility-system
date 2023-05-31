import os

# UNIVERSALLY USEFUL
SRC_PATH = os.path.dirname(os.path.realpath(__file__))
# the rest of the directory tree is assummed known
SIM_INPUTS_PATH = 'data/sim_in/'
SIM_OUTPUTS_PATH = 'data/sim_out/'

# SIMULATION DATA COLLECTION INPUTS
DATA_START_TIME = '05:00:00' # NOT EARLIER THAN 4AM
DATA_END_TIME = '21:00:00' # NOT LATER THAN 9PM
# GTFS_ZIP_FILE = '2022-10.zip'
# AVL_FILE = '2022-10_Route_81.csv'
# ODX_FILE = '2022-10_Route_81.csv'
# RUNS_FILE = 'v651_Route_81.csv'
# YEAR_MONTH = '2022-10' # this is for zip file name (sometimes it doesn't have any name)
# START_DATE = '2022-10-03'
# END_DATE = '2022-10-28'
# HOLIDAYS = ['2022-10-10'] 
# ROUTES = ['81']
GTFS_ZIP_FILE = '2022-11.zip'
AVL_FILE = '2022-1114-1209_JPark.csv'
ODX_FILE = '2022-1114-1209_JPark.csv'
RUNS_FILE = 'v652_JPark.csv'
START_DATE = '2022-11-14'
END_DATE = '2022-12-09'
HOLIDAYS = ['2022-11-24'] 

OUTBOUND_DIRECTIONS = {'81': 'East',
                       '91': 'South',
                       '92': 'East'}
INBOUND_DIRECTIONS = {'81': 'West',
                      '91': 'North',
                      '92': 'West'}
OUTBOUND_TERMINALS = {'81': (14102, 3773),
                      '91': (14103, 9617),
                      '92': (14108, 5425)} 
INBOUND_TERMINALS = {'81': (3773, 14102),
                      '91': (9617, 14103),
                      '92': (1038, 14108)} #92 shows 18579 ASK!!
ROUTES = ['81', '91', '92']

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
KEY_STOP_NAMES = {'81': ['Jefferson Park', 'Cicero', 'Pulaski', 'Kimball','Western'],
                  '91': [],
                  '92': []}

ANIMATION = {
    'y_lim': (41.952583,41.984450),
    'x_lim': (-87.768183,-87.642206),
    'text_loc': (-87.769,41.953)
}
