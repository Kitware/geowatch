import os  # NOQA
import sys  # NOQA

# Adds the "modules" subdirectory to the python path.
# See https://gitlab.kitware.com/smart/watch/-/merge_requests/148#note_1050127
# for discussion of how to refactor this in the future.
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
