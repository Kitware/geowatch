import pytest  # NOQA


def pytest_addoption(parser):
    # Allow CLI flags to be passed in as an option on sys.argv
    # Thes are used by xdoctest, pytest currently does nothing with them
    parser.addoption("--network", action="store_true")
    parser.addoption("--slow", action="store_true")
    parser.addoption("--gpu", action="store_true")
    parser.addoption("--customdirs", action="store_true")
