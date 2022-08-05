#!/bin/bash
__doc__="
This runs linting checks for the style this repo seeks to conform to.

This must be run from the repo directory.

Requirements:
    -r ./requirements/linting.txt
"
WATCH_LINT_DEFAULT_MODE=lint WATCH_LINT_DEFAULT_DPATH=watch python dev/lint.py "$@"
