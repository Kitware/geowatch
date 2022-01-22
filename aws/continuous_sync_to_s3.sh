#!/bin/bash
__doc__='
Continuously sync a directory to S3

Usage:


Ignore:
    LOCAL_DPATH=$HOME/data/work/toy_change/training
    ls $LOCAL_DPATH

    S3_ROOT=s3://kitware-smart-watch-data/
    S3_DPATH=$S3_ROOT/training

    tree $LOCAL_DPATH
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/
    aws s3 --profile iarpa ls $S3_ROOT/training/toothbrush/joncrall/ToyDataMSI/
'


sync_to_smart_s3(){
    __doc__="
    Does a single sync operation from a local directory to Kitware's
    SMART S3 Bucket in the 'training' subfolder.
    "
    LOCAL_DPATH=$1

    S3_ROOT=s3://kitware-smart-watch-data
    S3_DPATH=$S3_ROOT/training

    echo "
    LOCAL_DPATH='$LOCAL_DPATH'
    S3_ROOT='$S3_ROOT'
    S3_DPATH='$S3_DPATH'
    "

    aws s3 --profile iarpa sync \
        --exclude '*/monitor/train/batch/*' \
        --exclude '*/monitor/validate/batch/*' \
        --exclude '*/monitor/sanity_check/batch/*' \
        "$LOCAL_DPATH" "$S3_DPATH"
}


sync_forever(){
    __doc__="
    Run the sync command in an infinite loop
    "
    LOCAL_DPATH=$1
    while true 
    do
        sleep 
        sync_to_smart_s3 "$LOCAL_DPATH"
    done
}


# MAIN

sync_forever "$1"
