#!/bin/bash


cd /flash
git clone git@gitlab.kitware.com:smart/smart_data_dvc.git /flash/smart_data_dvc

# Lookup the location of the hard drive cache
HDD_CACHE_DIR=$(dvc cache dir)
HDD_CACHE_DIR=$(cd "$HOME"/data/dvc-repos/smart_data_dvc && dvc cache dir)
echo "HDD_CACHE_DIR = $HDD_CACHE_DIR"

cd /flash/smart_data_dvc
dvc remote add --local local_store "$HDD_CACHE_DIR"
dvc remote add --local namek ssh://namek/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache

dvc pull -r local_store annotations.dvc

cd /flash/smart_data_dvc/Drop6

SSD_DVC_DATA_DPATH=$(smartwatch_dvc --tags phase2_data --hardware=ssd)
HDD_DVC_DATA_DPATH=$(smartwatch_dvc --tags phase2_data --hardware=hdd)
echo "SSD_DVC_DATA_DPATH = $SSD_DVC_DATA_DPATH"
echo "HDD_DVC_DATA_DPATH = $HDD_DVC_DATA_DPATH"

# cd $HDD_DVC_DATA_DPATH/Drop6

python unpack.py \
    --src_bundle "$HDD_DVC_DATA_DPATH/Drop6" \
    --dst_bundle "$SSD_DVC_DATA_DPATH/Drop6" \
    --workers 8
