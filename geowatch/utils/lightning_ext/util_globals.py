"""
moved to geowatch.utils.util_globals
"""
from kwutil.util_eval import restricted_eval, RestrictedSyntaxError  # NOQA
from kwutil.util_resources import request_nofile_limits, check_shm_limits  # NOQA
from geowatch.utils.util_globals import coerce_num_workers, configure_global_attributes  # NOQA
