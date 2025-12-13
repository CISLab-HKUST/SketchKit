import os
import sys


IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
ASSETS_ROOT = f'{PROJECT_ROOT}/assets/'
DATA_ROOT = f'{ASSETS_ROOT}/data/'
CHECKPOINTS_ROOT = f'{ASSETS_ROOT}checkpoints/'
UI_RESOURCES = f'{ASSETS_ROOT}/ui_resources/'

MAX_VS = 100000