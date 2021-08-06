#!/usr/bin/env python
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Print hello world!")

    hello_world(**vars(parser.parse_known_args()[0]))


def hello_world():
    """
    Notice the doctest. It is run in our CI!

    Example:
        >>> from watch.cli.hello_world import *  # NOQA
        >>> hello_world()
    """
    print("hello world!")


if __name__ == '__main__':
    sys.exit(main())
