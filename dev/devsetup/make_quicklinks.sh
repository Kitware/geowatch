#!/bin/bash
__doc__='
Hardcoded symlink commands that populates a "quicklinks" directory. I typically
put this "quicklinks" folder in my favorites, so I can quickly navigate to
different locations on different machines.
'
mkdir -p  "$HOME"/quicklinks/

ln -sf  "$HOME"/remote/namek/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/smart_expt_dvc_dvc

ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_data_dvc-hdd "$HOME"/quicklinks/toothbrush_smart_data_dvc-hdd
ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd "$HOME"/quicklinks/toothbrush_smart_data_dvc-ssd
ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/toothbrush_smart_expt_dvc

ln -sf  "$HOME"/remote/ooo/data/dvc-repos/smart_data_dvc-ssd "$HOME"/quicklinks/ooo_smart_data_dvc-ssd

ln -sf  "$HOME"/remote/yardrat/data/dvc-repos/smart_data_dvc-ssd "$HOME"/quicklinks/yardrat_smart_data_dvc-ssd
ln -sf  "$HOME"/remote/yardrat/data/dvc-repos/smart_data_dvc-hdd "$HOME"/quicklinks/yardrat_smart_data_dvc-hdd
ln -sf  "$HOME"/remote/yardrat/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/yardrat_smart_expt_dvc


ln -sf  "$HOME"/remote/horologic/data/dvc-repos/smart_expt_dvc "$HOME"/quicklinks/horologic_smart_expt_dvc


ln -sf  "$HOME"/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall "$HOME"/quicklinks/toothbrush_training
