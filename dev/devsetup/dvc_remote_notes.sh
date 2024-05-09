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


# Fixup permissions while avoiding DVC issues


# Give all real directories (not symlinks) all group permissions and set the sticky bit
sudo fdfind --hidden --no-ignore --type directory --exec chmod g+rwxs

# Give all real files (not symlinks) group readwrite permission, exclude the
# .dvc cache directory, which needs special permissions
sudo fdfind --hidden --no-ignore --type file --exclude "**/cache/files/md5/*" --exec chmod g+rw


# Give all real directories (not symlinks) all group permissions and set the sticky bit
sudo fdfind -uu -t d -x chmod g+rwxs
sudo fdfind -uu -t f -E "**/cache/files/md5/*" -x chmod g+rw


# Ensure good permissions on new data drives
MOUNT_OWNER=root
MOUNT_GROUP=smart
MOUNT_DPATH=/data2
# Reset the owner on the mounted filesystem
sudo chown "$MOUNT_OWNER":"$MOUNT_GROUP" "$MOUNT_DPATH"
# Set group and user permissions to be permissive
# Restrict other permissions
sudo chmod ug+srwx $MOUNT_DPATH
sudo chmod o-rwx $MOUNT_DPATH
# Set file access control lists (ACL) so new directories and files are group read/write by default
# https://unix.stackexchange.com/questions/12842/make-all-new-files-in-a-directory-accessible-to-a-group
# Note: not all filesystems support ACL
sudo setfacl -d -m "group:${MOUNT_GROUP}:rwx" "$MOUNT_DPATH"
sudo setfacl -m "group:${MOUNT_GROUP}:rwx" "$MOUNT_DPATH"



# Moving data to new directory
rsync -avprRP /data/projects/smart/smart_phase3_data/.dvc/./cache /data2/projects/smart/smart_phase3_data/.dvc




cd /data/joncrall/dvc-repos/smart_phase3_expt
DPATH=$(pwd)
DISK_TYPE=$(python -m geowatch.cli.experimental.disk_info --key hwtype)
geowatch_dvc add --name="smart_phase3_expt-$DISK_TYPE" --path="$DPATH" --hardware="$DISK_TYPE" --tags=phase3_expt

cd /data/joncrall/dvc-repos/smart_expt
DPATH=$(pwd)
DISK_TYPE=$(python -m geowatch.cli.experimental.disk_info --key hwtype)
geowatch_dvc add --name="smart_phase2_expt-$DISK_TYPE" --path="$DPATH" --hardware="$DISK_TYPE" --tags=phase2_expt


cd /data/joncrall/dvc-repos/smart_phase3_data
DPATH=$(pwd)
DISK_TYPE=$(python -m geowatch.cli.experimental.disk_info --key hwtype)
geowatch_dvc add --name="smart_phase3_data-$DISK_TYPE" --path="$DPATH" --hardware="$DISK_TYPE" --tags=phase3_data

cd /media/joncrall/flash1/smart_phase3_data
DPATH=$(pwd)
DISK_TYPE=$(python -m geowatch.cli.experimental.disk_info --key hwtype)
geowatch_dvc add --name="smart_phase3_data-$DISK_TYPE" --path="$DPATH" --hardware="$DISK_TYPE" --tags=phase3_data

cd /media/joncrall/flash1/smart_phase3_data
DPATH=$(pwd)
DISK_TYPE=$(python -m geowatch.cli.experimental.disk_info --key hwtype)
geowatch_dvc add --name="smart_phase3_data-$DISK_TYPE" --path="$DPATH" --hardware="$DISK_TYPE" --tags=phase3_data
