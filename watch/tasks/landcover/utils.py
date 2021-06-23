import datetime
import logging
import os
import sys


def setup_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)-40s - %(levelname)-7s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.INFO)

