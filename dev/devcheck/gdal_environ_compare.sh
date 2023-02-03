#!/bin/bash
__doc__='

Notes:
    https://gis.stackexchange.com/questions/89444/file-size-inflation-normal-with-gdalwarp/89549#89549
    https://gis.stackexchange.com/questions/372786/speed-up-gdalwarp

Input is 37750,34015

### Recommended by TnE via David Joy
export CPL_CURL_VERBOSE=NO
export CPL_TMPDIR=/local/work
export CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES
export GDAL_CACHEMAX=1024
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_CONNECTTIMEOUT=10
export GDAL_HTTP_MAX_RETRY=10
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_RETRY_DELAY=10
export GDAL_HTTP_TIMEOUT=60
export GDAL_INGESTED_BYTES_AT_OPEN=32000
export GDAL_SWATH_SIZE=200000000
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE="$(xdev pint "10 megabytes" "bytes" --precision=0)"
export VSI_CACHE_SIZE=100000000


'

export AWS_DEFAULT_PROFILE=iarpa
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR


#export CPL_CURL_VERBOSE=NO
#export CPL_TMPDIR=/local/work
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES
export GDAL_CACHEMAX=1024
export GDAL_HTTP_CONNECTTIMEOUT=10
export GDAL_HTTP_MAX_RETRY=10
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_RETRY_DELAY=10
export GDAL_HTTP_TIMEOUT=60
export GDAL_INGESTED_BYTES_AT_OPEN=32000
export GDAL_SWATH_SIZE=200000000
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=100000000 

time gdalwarp -overwrite \
    -te 8.420848 47.297216 8.581097 47.467417 -te_srs epsg:4326 \
    -t_srs epsg:32632 -tr 30 30 \
    -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE \
    /vsis3/smart-data-accenture/ta-1/ta1-wv-acc-2/32/T/MT/2018/3/24/18MAR24105605-P1BS-501687882040_02_P006/18MAR24105605-P1BS-501687882040_02_P006_ACC_QA.tif \
    tmp.tif

#real	5m40.106s
#user	0m4.702s
#sys	0m6.488s

export AWS_DEFAULT_PROFILE=iarpa
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR


#export CPL_CURL_VERBOSE=NO
#export CPL_TMPDIR=/local/work
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES
#export GDAL_CACHEMAX=1024
export GDAL_CACHEMAX=2048
export GDAL_HTTP_CONNECTTIMEOUT=10
export GDAL_HTTP_MAX_RETRY=10
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_RETRY_DELAY=10
export GDAL_HTTP_TIMEOUT=60
export GDAL_INGESTED_BYTES_AT_OPEN=32000
export GDAL_SWATH_SIZE=200000000
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=100000000 

time gdalwarp -overwrite \
    -te 8.420848 47.297216 8.581097 47.467417 -te_srs epsg:4326 \
    -t_srs epsg:32632 \
    -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE -co NUM_THREADS=16 \
    /vsis3/smart-data-accenture/ta-1/ta1-wv-acc-2/32/T/MT/2018/3/24/18MAR24105605-P1BS-501687882040_02_P006/18MAR24105605-P1BS-501687882040_02_P006_ACC_QA.tif \
    tmp_rawres.tif
#real	15m36.874s
#user	0m24.162s
#sys	0m10.720s


 #-tr 2 2

time gdalwarp -overwrite -multi --debug off \
    -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE \
    -co NUM_THREADS=20 --config GDAL_CACHEMAX 1500 \
    /vsis3/smart-data-accenture/ta-1/ta1-wv-acc-2/32/T/MT/2018/3/24/18MAR24105605-P1BS-501687882040_02_P006/18MAR24105605-P1BS-501687882040_02_P006_ACC_QA.tif \
    tmp_orig.tif

#real	15m30.879s
#user	0m29.134s
#sys	0m12.905s


# First 5 minutes only pulled down - 276K	tmp_orig.tif
# Network is running at 2.5 MiB/s
# At 7 minutes - 340K
# At 29 minutes had - 19M

    #-t_srs epsg:32632 -of COG -te 8.420848 47.297216 8.581097 47.467417 \
    #-te_srs epsg:4326 -wm 1500 \
