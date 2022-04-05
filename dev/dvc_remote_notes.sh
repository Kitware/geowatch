REMOTE=toothbrush
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

REMOTE=namek
dvc remote add $REMOTE ssh://$REMOTE/data/dvc-repos/smart_watch_dvc/.dvc/cache

cd Cropped

dvc pull -R . -r toothbrush
dvc pull -R . -r namek
