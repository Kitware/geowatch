#!/bin/bash

if [ -d "$HOME"/code/metrics-and-test-framework ]; then
    git clone git@gitlab.kitware.com:smart/metrics-and-test-framework.git "$HOME"/code/metrics-and-test-framework
fi

pip install -e "$HOME"/code/metrics-and-test-framework
