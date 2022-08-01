#!/bin/bash
__doc__="
This runs linting checks for the style this repo seeks to conform to.

This must be run from the repo directory.

Requirements:
    -r ./requirements/linting.txt
"
python dev/lint.py watch --mode=lint
