REMOTE=toothbrush
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

REMOTE=namek
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

cd Cropped

dvc pull -R . -r toothbrush
dvc pull -R . -r namek



### Data Remotes

# See: ~/code/watch/dev/devsetup/dvc_update_remotes.py

dvc remote add -f horologic     ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
dvc remote add -f horologic_hdd ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
dvc remote add -f horologic_ssd ssh://horologic.kitware.com/flash/smart_data_dvc/.dvc/cache

dvc remote add -f namek     ssh://namek/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache
dvc remote add -f namek_hdd ssh://namek/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/.dvc/cache
dvc remote add -f namek_ssd ssh://namek/flash/smart_data_dvc/.dvc/cache

dvc remote add -f toothbrush     ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-hdd/.dvc/cache
dvc remote add -f toothbrush_hdd ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-hdd/.dvc/cache
dvc remote add -f toothbrush_ssd ssh://toothbrush/data/joncrall/dvc-repos/smart_data_dvc-ssd/.dvc/cache

dvc remote add -f ooo     ssh://ooo/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache -f
dvc remote add -f ooo_hdd ssh://ooo/data/joncrall/dvc-repos/smart_data_dvc/.dvc/cache -f
dvc remote add -f ooo_ssd ssh://ooo/flash/smart_data_dvc/.dvc/cache -f







### Expt Remotes

dvc remote add --local toothbrush ssh://toothbrush/data/joncrall/dvc-repos/smart_expt_dvc/.dvc/cache


# On horologic
dvc remote add --local local_store /data/dvc-caches/smart_watch_dvc


### See ALso:
"$HOME/data/dvc-repos/smart_data_dvc/Drop6/unpack.py"
