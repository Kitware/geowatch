"""
moved to watch.utils.util_globals
"""
from watch.utils.util_eval import restricted_eval, RestrictedSyntaxError  # NOQA
from watch.utils.util_resources import request_nofile_limits, check_shm_limits  # NOQA
from watch.utils.util_globals import coerce_num_workers, configure_global_attributes  # NOQA
