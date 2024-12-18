"""
Currently just an alias for mtm/attach_features.
"""
import sys
from geowatch.tasks.rutgers_material_seg_v2.mtm.attach_features import predict as main
from geowatch.tasks.rutgers_material_seg_v2.mtm.attach_features import __cli__  # NOQA


if __name__ == '__main__':
    main(sys.argv[1:])
