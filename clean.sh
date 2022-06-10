#!/bin/bash
__doc__="
Remove common temp files that get written to this directory
"

rm -rf text_logs.log
rm -rf lightning_logs
rm -rf __pycache__
rm -rf htmlcov
rm -rf dist
rm -rf pred_out

CLEAN_PYTHON='find . -iname *.pyc -delete ; find . -iname *.pyo -delete ; find . -regex ".*\(__pycache__\|\.py[co]\)" -delete'
bash -c "$CLEAN_PYTHON"
rm -rf .coverage
