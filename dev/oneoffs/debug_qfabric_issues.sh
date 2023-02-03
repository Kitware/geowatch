python -m watch reproject_annotations --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imgonly-BLA_QFABRIC_R004.kwcoco.json \
    --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imganns-BLA_QFABRIC_R004.kwcoco.json \
    '--site_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/site_models/*.geojson' \
    --region_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R004.geojson


python -m watch reproject_annotations --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imgonly-BLA_QFABRIC_R119.kwcoco.json --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imganns-BLA_QFABRIC_R119.kwcoco.json '--site_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/site_models/*.geojson' --region_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R119.geojson                                                                                                                                   


python -m watch reproject_annotations --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imgonly-BLA_QFABRIC_R199.kwcoco.json --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imganns-BLA_QFABRIC_R199.kwcoco.json '--site_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/site_models/*.geojson' --region_models=/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R199.geojson                                                                                                                                   

AWS_PROFILE=iarpa python -m watch.cli.coco_align_geotiffs --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Uncropped-QFabric-c30-worldview-nitf/data_BLA_QFABRIC_R138_fielded.kwcoco.json --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c30-worldview-nitf/imgonly-BLA_QFABRIC_R138.kwcoco.json --regions /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R138.geojson --context_factor=1 --geo_preprop=auto --keep=img --force_nodata=None --include_channels=None --exclude_channels=None --visualize=False --debug_valid_regions=True --rpc_align_method orthorectify --verbose=0 --aux_workers=0 --target_gsd=2 --workers=12


AWS_PROFILE=iarpa gdalinfo /vsis3/smart-imagery/worldview-nitf/50/R/QP/2019/04/08/19APR08030939-M1BS-015048382010_01_P001/19APR08030939-M1BS-015048382010_01_P001.NTF

AWS_PROFILE=iarpa gdalwarp -overwrite -multi --debug off -t_srs epsg:32650 -of COG -te 119.21518526629589 26.031683444072968 119.21548391370007 26.04946025426622 -te_srs epsg:4326 -wm 1500 -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE -co NUM_THREADS=2 --config GDAL_CACHEMAX 1500 /vsis3/smart-imagery/worldview-nitf/50/R/QP/2019/04/08/19APR08030939-M1BS-015048382010_01_P001/19APR08030939-M1BS-015048382010_01_P001.NTF /tmp/.tmpwarp.tmp2pw352b3.tif

AWS_PROFILE=iarpa gdalwarp -overwrite -rpc -multi --debug off -t_srs epsg:32650 -of COG -te 119.21518526629589 26.031683444072968 119.21588391370007 26.04946025426622 -te_srs epsg:4326 -et 0 -wm 1500 -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE -co NUM_THREADS=2 -to RPC_DEM=/home/local/KHQ/jon.crall/.cache/watch/girder/gtop30/gt30e100n40.tif --config GDAL_CACHEMAX 1500 /vsis3/smart-imagery/worldview-nitf/50/R/QP/2018/01/12/18JAN12025038-M1BS-015048392010_01_P001/18JAN12025038-M1BS-015048392010_01_P001.NTF /tmp/.tmpwarp.tmpzud4thh5.tif


 /home/local/KHQ/jon.crall
 BLA_QFABRIC_R138.geojson
 cd $HOME/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/


AWS_PROFILE=iarpa python -m watch.cli.coco_align_geotiffs --src /home/joncrall/data/dvc-repos/smart_data_dvc/public/Uncropped-QFabric-c80-worldview-nitf/data_BLA_QFABRIC_R089_fielded.kwcoco.json --dst /home/joncrall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c80-worldview-nitf/imgonly-BLA_QFABRIC_R089.kwcoco.json --regions /home/joncrall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R089.geojson --context_factor=1 --geo_preprop=auto --keep=img --force_nodata=None --include_channels=None --exclude_channels=None --visualize=False --debug_valid_regions=False --rpc_align_method orthorectify --verbose=0 --aux_workers=0 --target_gsd=2 --workers=12


python -m watch reproject_annotations --src /home/joncrall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c80-worldview-nitf/imgonly-BLA_QFABRIC_R129.kwcoco.json --dst /home/joncrall/data/dvc-repos/smart_data_dvc/public/Aligned-QFabric-c80-worldview-nitf/imganns-BLA_QFABRIC_R129.kwcoco.json '--site_models=/home/joncrall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/site_models/*.geojson' --region_models=/home/joncrall/data/dvc-repos/smart_data_dvc/public/annotations-qfabric/orig/region_models/BLA_QFABRIC_R129.geojson
