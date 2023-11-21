import os
import logging


def get_python_logger():
    return logging.getLogger("lightning")


def setup_python_logging(log_dir):
    """Adds logging to the console and puts it in the tensorboard
    logging directory."""

    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, 'console.log')

    console_log = get_python_logger()
    handler = logging.FileHandler(filename, mode='w')
    fmt = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)
    console_log.addHandler(handler)
