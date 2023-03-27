#!/bin/bash
mkdir -p  "$HOME"/quicklinks/

ln -sf  "$HOME"/remote/namek/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/namek_smart_data_dvc

ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_data_dvc-hdd "$HOME"/quicklinks/toothbrush_smart_data_dvc-hdd
ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd "$HOME"/quicklinks/toothbrush_smart_data_dvc-ssd
ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/toothbrush_smart_expt_dvc
