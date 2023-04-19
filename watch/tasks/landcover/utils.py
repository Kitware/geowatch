import logging


def setup_logging(level=logging.WARNING):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)-40s - %(levelname)-7s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.INFO)
