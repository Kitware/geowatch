# !/bin/bash
# NOTE: Proof of concept for warp from S3:

aws s3 --profile iarpa ls s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/

gdalinfo \
    --config AWS_DEFAULT_PROFILE "iarpa" \
    "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif"

gdalwarp \
    --config AWS_DEFAULT_PROFILE "iarpa" \
    -te_srs epsg:4326 \
    -te -81.51 29.99 -81.49 30.01 \
    -t_srs epsg:32617 \
    -overwrite \
    -of COG \
    -co OVERVIEWS=AUTO \
    "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif" \
    partial_crop2.tif
gdalinfo partial_crop2.tif
kwplot partial_crop2.tif

gdalinfo \
    --config AWS_DEFAULT_PROFILE "iarpa" \
    --config AWS_CONFIG_FILE "$HOME/.aws/config" \
    --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
    "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif"

gdalwarp \
    --config AWS_DEFAULT_PROFILE "iarpa" \
    --config AWS_CONFIG_FILE "$HOME/.aws/config" \
    --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
    -te_srs epsg:4326 \
    -te -43.51 -23.01 -43.49 -22.99 \
    -t_srs epsg:32723 \
    -overwrite \
    -of COG \
    "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif" \
    partial_crop.tif
kwplot partial_crop.tif


"""
    Ignore:
        in_fpath =
        s3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2

        gdalwarp 's3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2' foo.tif

    aws s3 --profile iarpa cp s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif foo.tif

    gdalwarp 's3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif' bar.tif
"""
