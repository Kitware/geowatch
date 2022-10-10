"""
Currently just an alias for kwcoco_to_geojson. Eventually, most of that logic
should move here.
"""
import sys
from watch.cli.kwcoco_to_geojson import main


if __name__ == '__main__':
    main(sys.argv[1:])
