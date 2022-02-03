#!/bin/bash
__doc__='
Continuously sync a directory to S3

Usage:
    # To run in the background you can do something likes minimal test

    LOCAL_DPATH="$HOME/tmp/test-s3-sync"
    mkdir -p $LOCAL_DPATH
    cd $LOCAL_DPATH
    source ~/code/watch/aws/smartwatch_s3_sync.sh && smartwatch_s3_sync_forever_in_tmux "$LOCAL_DPATH"

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/

Ignore:
    LOCAL_DPATH=$HOME/data/work/toy_change/training
    ls $LOCAL_DPATH

    S3_ROOT=s3://kitware-smart-watch-data/
    S3_DPATH=$S3_ROOT/training

    tree $LOCAL_DPATH
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/
    aws s3 --profile iarpa ls $S3_ROOT/sync_root/toothbrush/joncrall/ToyDataMSI/
'


smartwatch_s3_sync_single(){
    __doc__="
    Does a single sync operation from a local directory to Kitware's
    SMART S3 Bucket in the 'training' subfolder.
    "
    LOCAL_DPATH=$1

    S3_ROOT=s3://kitware-smart-watch-data
    S3_DPATH_ROOT=$S3_ROOT/sync_root

    if [[ "$HOSTNAME" == "" ]]; then 
        export HOSTNAME="unknown"
    fi

    PART2=$HOSTNAME/$USER

    # TODO: respect the "./" syntax like rsync, for now lets just hack it
    #STR="/fds/fds/./fd33s"
    #PART1="${STR%/./*}"
    #PART2="${STR#*/./}"
    #echo "PART1 = $PART1"
    #echo "PART2 = $PART2"
    ##PART1=$(echo /fds/fds/./fd33s | cut -f1 -d'./')
    ##echo "PART1 = $PART1"
    ##PART2=$(echo $STR | cut -f2 -d'./')

    if [[ "$PART2" != "" ]]; then
        S3_DPATH_FULL=$S3_DPATH_ROOT/$PART2/
    else
        S3_DPATH_FULL=$S3_DPATH_ROOT/
    fi

    echo "
    LOCAL_DPATH='$LOCAL_DPATH'
    S3_ROOT='$S3_ROOT'
    S3_DPATH_ROOT='$S3_DPATH_ROOT'
    S3_DPATH_FULL='$S3_DPATH_FULL'
    "
    #aws s3 --profile iarpa sync \
    aws s3 sync \
        "$LOCAL_DPATH" "$S3_DPATH_FULL"
}


smartwatch_s3_sync_forever(){
    __doc__="
    Run the sync command in an infinite loop
    "
    LOCAL_DPATH=$1
    echo "LOCAL_DPATH = $LOCAL_DPATH"
    while true 
    do
        sleep 1.0
        smartwatch_s3_sync_single "$LOCAL_DPATH"
    done
}


smartwatch_s3_sync_forever_in_tmux(){
    LOCAL_DPATH=$1
    echo "LOCAL_DPATH = $LOCAL_DPATH"
    tmux new-session -d -s sync_worker1 "source $HOME/code/watch/aws/smartwatch_s3_sync.sh && smartwatch_s3_sync_forever $LOCAL_DPATH"
    tmux ls
}
