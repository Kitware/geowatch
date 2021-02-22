#!/usr/bin/env python

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Print hello world!")

    hello_world(**vars(parser.parse_args()))


def hello_world():
    print("hello world!")


if __name__ == '__main__':
    sys.exit(main())
