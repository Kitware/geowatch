
python -m watch.cli.stac_search \
        --region_file "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/annotations/region_models/US_C011.geojson" \
        --search_json "auto" \
        --cloud_cover "10" \
        --sensors "L2-S2-L8" \
        --api_key "env:SMART_STAC_API_KEY" \
        --max_products_per_region "None" \
        --max_products_per_region "None" \
        --mode area \
        --verbose 2 \
        --outfile "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/_query/items/US_C011.input"



python -m watch.cli.baseline_framework_ingress \
        --aws_profile iarpa \
        --jobs avail \
        --virtual --requester_pays \
        --outdir "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/ingress" \
        --catalog_fpath "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/ingress/catalog_US_C011.json" \
        "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/_query/items/US_C011.input"


AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' python -m watch.cli.ta1_stac_to_kwcoco \
        "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/ingress/catalog_US_C011.json" \
        --outpath="/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/data_US_C011.kwcoco.json" \
        --ignore_duplicates \
        --jobs "8"

AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' python -m watch.cli.coco_add_watch_fields \
        --src "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/data_US_C011.kwcoco.json" \
        --dst "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Uncropped-Drop4-2022-07-18-c10-L2-S2-L8/data_US_C011_fielded.kwcoco.json" \
        --enable_video_stats=False \
        --overwrite=warp \
        --target_gsd=10 \
        --remove_broken=True \
        --workers="0"



https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/NQ/2017/6/S2A_23KNQ_20170618_0_L2A/B05.tif

AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' gdalinfo /vsis3/sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/NQ/2017/6/S2A_23KNQ_20170618_0_L2A/B05.tif

gdalinfo /vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/PQ/2019/6/S2B_23KPQ_20190623_0_L2A/B09.tif




AWS_DEFAULT_PROFILE=iarpa gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-wv-acc/14/T/QL/2014/9/29/14SEP29174805-M1BS-014484503010_01_P003_ACC/14SEP29174805-M1BS-014484503010_01_P003_ACC_B05.tif --config CPL_LOG image.log

AWS_DEFAULT_PROFILE=iarpa gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-ls-acc/32/T/NT/2021/1/11/LC08_L1TP_194027_20210111_20210307_02_T1_ACC/LC08_L1TP_194027_20210111_20210307_02_T1_ACC_B01.tif

AWS_DEFAULT_PROFILE=iarpa gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/43/R/GM/2020/2/14/S2A_43RGM_20200214_0_L1C_ACC/S2A_43RGM_20200214_0_L1C_ACC_B08.tif

AWS_DEFAULT_PROFILE=iarpa gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-pd-acc/19/K/BA/2017/10/29/20171029_184057_0c43_3B_AnalyticMS_ACC/20171029_184057_0c43_3B_AnalyticMS_ACC_B03.tif



__doc__='
config = {
    "absolute": False,
    "dst": "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/data.kwcoco.json",
    "src": ["/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C000.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-NZ_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-RU_C000.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CN_C001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C003.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CH_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-PE_C001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R005.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R004.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R002.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C011.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C012.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R006.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-PE_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R005.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C002.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BH_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R002.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C002.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R004.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-LT_R001.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R007.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-ET_C000.kwcoco.json",
    "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C010.kwcoco.json"],
}
reading datasets  0/31... rate=0 Hz, eta=?, total=0:00:00reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C000.kwcoco.json"
reading datasets  1/31... rate=42.89 Hz, eta=0:00:00, total=0:00:00reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-NZ_R001.kwcoco.json"
reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-RU_C000.kwcoco.json"
reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CN_C001.kwcoco.json"
reading datasets  4/31... rate=1.29 Hz, eta=0:00:20, total=0:00:03reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R001.kwcoco.json"
reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C003.kwcoco.json"
reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C001.kwcoco.json"
reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CH_R001.kwcoco.json"
reading datasets  8/31... rate=1.75 Hz, eta=0:00:13, total=0:00:04reading fpath = "/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json"
ERROR ex = Exception("Specified fpath=/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json does not exist. If you are trying to create a new dataset fist create a CocoDataset without any arguments, and then set the fpath attribute. We may loosen this requirement in the future.")
Traceback (most recent call last):
  File "/home/joncrall/.pyenv/versions/3.10.5/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/joncrall/.pyenv/versions/3.10.5/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/joncrall/code/kwcoco/kwcoco/__main__.py", line 9, in <module>
    main()
  File "/home/joncrall/code/kwcoco/kwcoco/cli/__main__.py", line 115, in main
    ret = main(cmdline=False, **kw)
  File "/home/joncrall/code/kwcoco/kwcoco/cli/coco_union.py", line 54, in main
    dset = kwcoco.CocoDataset.coerce(fpath)
  File "/home/joncrall/code/kwcoco/kwcoco/coco_dataset.py", line 1001, in coerce
    self = kwcoco.CocoDataset(dset_fpath, **kw)
  File "/home/joncrall/code/kwcoco/kwcoco/coco_dataset.py", line 5121, in __init__
    raise Exception(ub.paragraph(
Exception: Specified fpath=/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json does not exist. If you are trying to create a new dataset fist create a CocoDataset without any arguments, and then set the fpath attribute. We may loosen this requirement in the future.
(pyenv3.10.5) joncrall@namek:~/code/watch$
'

__doc__='

P001_ACC_B01.tif                                                                                                                                                         [1293/1441]
    "name": "20APR05161504-P1BS-014887566010_01_P001_ACC",
    "sensor_coarse": "WV",
}
collect jobs 353/353... rate=3.30 Hz, eta=0:00:00, total=0:01:14
Wrote: /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/Uncropped-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_BR_R005_0127_box.kwcoco.json
++ AWS_DEFAULT_PROFILE=iarpa
++ python -m watch.cli.coco_add_watch_fields --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/Uncropped-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_BR_R005_0127_box.kwcoco.
json --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/Uncropped-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_BR_R005_0127_box_fielded.kwcoco.json --enable_video_stats=False
--overwrite=warp --target_gsd=4 --remove_broken=True --workers=20
"
config = {
    "default_gsd": None,
    "dst": "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/Uncropped-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_BR_R005_0127_box_fielded.kwcoco.json",
    "edit_geotiff_metadata": False,
    "enable_intensity_stats": False,
    "enable_valid_region": False,
    "enable_video_stats": False,
    "mode": "process",
    "overwrite": "warp",
    "remove_broken": True,
    "src": "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/Uncropped-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_BR_R005_0127_box.kwcoco.json",
    "target_gsd": 4,
    "workers": "20",
}
read dataset
dset = <CocoDataset(tag=data_BR_R005_0127_box.kwcoco.json, n_anns=0, n_imgs=346, n_videos=0, n_cats=0) at 0x7ff8644e2500>
start populate
submit populate imgs 346/346... rate=764.17 Hz, eta=0:00:00, total=0:00:00
collect populate imgs 252/346... rate=9.24 Hz, eta=0:00:10, total=0:00:27ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P
001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not recognized as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not recogniz
ed as a supported file format.
collect populate imgs 264/346... rate=9.44 Hz, eta=0:00:08, total=0:00:27ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P
001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not recognized as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not recogniz
ed as a supported file format.
ex=RuntimeError("`/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not
recognized as a supported file format. for /vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_
01_P001_ACC_B01.tif")
ex=`/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not recognized as
a supported file format. for /vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B0
1.tif
ex.__dict__={}
concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "/home/local/KHQ/jon.crall/.pyenv/versions/3.10.5/lib/python3.10/concurrent/futures/process.py", line 246, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "/home/local/KHQ/jon.crall/code/watch/watch/utils/kwcoco_extensions.py", line 334, in coco_populate_geo_img_heuristics2
    errors = _populate_canvas_obj(
  File "/home/local/KHQ/jon.crall/code/watch/watch/utils/kwcoco_extensions.py", line 520, in _populate_canvas_obj
    info = watch.gis.geotiff.geotiff_metadata(
  File "/home/local/KHQ/jon.crall/code/watch/watch/gis/geotiff.py", line 52, in geotiff_metadata
    ref = util_gdal.GdalDataset.open(gpath, "r", virtual_retries=3)
  File "/home/local/KHQ/jon.crall/code/watch/watch/utils/util_gdal.py", line 1059, in open
    raise RuntimeError(msg + f" for {_path}")
RuntimeError: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not rec
ognized as a supported file format. for /vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_
P001_ACC_B01.tif
"""
The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/local/KHQ/jon.crall/.pyenv/versions/3.10.5/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/local/KHQ/jon.crall/.pyenv/versions/3.10.5/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/local/KHQ/jon.crall/code/watch/watch/cli/coco_add_watch_fields.py", line 153, in <module>
    main()
  File "/home/local/KHQ/jon.crall/code/watch/watch/cli/coco_add_watch_fields.py", line 122, in main
    kwcoco_extensions.populate_watch_fields(dset, **populate_kw)
  File "/home/local/KHQ/jon.crall/code/watch/watch/utils/kwcoco_extensions.py", line 163, in populate_watch_fields
    coco_populate_geo_heuristics(
  File "/home/local/KHQ/jon.crall/code/watch/watch/utils/kwcoco_extensions.py", line 239, in coco_populate_geo_heuristics
    img = job.result()
  File "/home/local/KHQ/jon.crall/.pyenv/versions/3.10.5/lib/python3.10/concurrent/futures/_base.py", line 439, in result
    return self.__get_result()
  File "/home/local/KHQ/jon.crall/.pyenv/versions/3.10.5/lib/python3.10/concurrent/futures/_base.py", line 391, in __get_result
    raise self._exception
RuntimeError: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif" not rec
ognized as a supported file format. for /vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_
P001_ACC_B01.tif
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144015-P1BS-014892948010_01_P006_ACC/15JUL10144015-P1BS-014892948010_01_P006_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144015-P1BS-014892948010_01_P006_ACC/15JUL10144015-P1BS-014892948010_01_P006_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144015-P1BS-014892948010_01_P006_ACC/15JUL10144015-P1BS-014892948010_01_P006_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144015-P1BS-014892948010_01_P006_ACC/15JUL10144015-P1BS-014892948010_01_P006_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133254-P1BS-014900691010_01_P001_ACC/14AUG23133254-P1BS-014900691010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133254-P1BS-014900691010_01_P001_ACC/14AUG23133254-P1BS-014900691010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133254-P1BS-014900691010_01_P001_ACC/14AUG23133254-P1BS-014900691010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133336-P1BS-014900692010_01_P001_ACC/14AUG23133336-P1BS-014900692010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133336-P1BS-014900692010_01_P001_ACC/14AUG23133336-P1BS-014900692010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133254-P1BS-014900691010_01_P001_ACC/14AUG23133254-P1BS-014900691010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133336-P1BS-014900692010_01_P001_ACC/14AUG23133336-P1BS-014900692010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/8/23/14AUG23133336-P1BS-014900692010_01_P001_ACC/14AUG23133336-P1BS-014900692010_01_P001_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144029-P1BS-014892990010_01_P004_ACC/15JUL10144029-P1BS-014892990010_01_P004_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144029-P1BS-014892990010_01_P004_ACC/15JUL10144029-P1BS-014892990010_01_P004_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144029-P1BS-014892990010_01_P004_ACC/15JUL10144029-P1BS-014892990010_01_P004_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2015/7/10/15JUL10144029-P1BS-014892990010_01_P004_ACC/15JUL10144029-P1BS-014892990010_01_P004_ACC_B01.tif" not recogni
zed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/7/4/14JUL04131637-P1BS-014900907010_01_P004_ACC/14JUL04131637-P1BS-014900907010_01_P004_ACC_B01.tif" not recogniz
ed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/7/4/14JUL04131637-P1BS-014900907010_01_P004_ACC/14JUL04131637-P1BS-014900907010_01_P004_ACC_B01.tif" not recogniz
ed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/7/4/14JUL04131637-P1BS-014900907010_01_P004_ACC/14JUL04131637-P1BS-014900907010_01_P004_ACC_B01.tif" not recogniz
ed as a supported file format.
ERROR 4: `/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2014/7/4/14JUL04131637-P1BS-014900907010_01_P004_ACC/14JUL04131637-P1BS-014900907010_01_P004_ACC_B01.tif" not recogniz
ed as a supported file format.
++ python -m watch.cli.stac_search --region_file /home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc/subregions/AE_R001_0266_box.geojson --search_json auto --cloud_cover 40 --s
'




aws s3 --profile=iarpa ls s3://smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC

/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/23/K/LQ/2016/7/1/16JUL01163115-P1BS-014893019010_01_P001_ACC/16JUL01163115-P1BS-014893019010_01_P001_ACC_B01.tif

cd "$HOME"/data/dvc-repos/smart_data_dvc/mwe
cp "$HOME"/data/dvc-repos/smart_data_dvc/subregions/BR_R005_0127_box.geojson query_region_BR_R005_0127_box.geojson

python -m watch.cli.stac_search \
    --region_file "query_region_BR_R005_0127_box.geojson" \
    --search_json "auto" \
    --cloud_cover "40" \
    --sensors "TA1-S2-WV-PD-ACC" \
    --api_key "env:SMART_STAC_API_KEY" \
    --max_products_per_region "None" \
    --mode area \
    --verbose 2 \
    --outfile "./mwe_big.input"

cat ./mwe_big.input | grep 16JUL01163115 > mwe.input

python -m watch.cli.baseline_framework_ingress \
    --aws_profile iarpa \
    --jobs 0 \
    --virtual \
    --outdir "./ingress" \
    --catalog_fpath "./catalog_mwe.json" \
    "mwe.input"

AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.ta1_stac_to_kwcoco \
    "./catalog_mwe.json" \
    --outpath="./mwe.kwcoco.json" \
    --from-collated --ignore_duplicates \
    --jobs "8"


# PREPARE Uncropped datasets
AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.coco_add_watch_fields \
    --src "./mwe.kwcoco.json" \
    --dst "./mwe_fielded.kwcoco.json" \
    --enable_video_stats=False \
    --overwrite=warp \
    --target_gsd=4 \
    --remove_broken=True \
    --workers="20"


AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.coco_align_geotiffs \
    --src "./mwe_fielded.kwcoco.json" \
    --dst "./mwe_aligned.kwcoco.json" \
    --regions "query_region_BR_R005_0127_box.geojson" \
    --context_factor=1 \
    --geo_preprop=auto \
    --keep=None \
    --force_nodata=-9999 \
    --include_channels="None" \
    --exclude_channels="None" \
    --visualize=False \
    --debug_valid_regions=False \
    --rpc_align_method affine_warp \
    --verbose=0 \
    --aux_workers=0 \
    --target_gsd=4 \
    --workers=12




##### DEBUG
#--region_file "$HOME/data/dvc-repos/smart_data_dvc/subregions/AE_R001_0198_box.geojson"  \
#--region_file "$HOME/data/dvc-repos/smart_data_dvc/annotations/region_models/AE_R001.geojson"  \

watch_env
python -m watch.cli.stac_search \
    --region_file "$HOME/data/dvc-repos/smart_data_dvc/annotations/region_models/AE_R001.geojson"  \
    --search_json "auto"  \
    --cloud_cover "40" \
    --sensors "TA1-S2-WV-PD-ACC" \
    --api_key "env:SMART_STAC_API_KEY"  \
    --max_products_per_region "None"  \
    --mode area  \
    --verbose 2 \
    --outfile ./test_results.input


__pycheck__="
import pystac_client
headers = {
    'x-api-key': os.environ['SMART_STAC_API_KEY']
}
provider = 'https://api.smart-stac.com'
catalog = pystac_client.Client.open(provider, headers=headers)
list(catalog.get_collections())

# Check that planet items exist
item_search = catalog.search(collections=['ta1-pd-acc'])
itemgen = item_search.items()
item = next(itemgen)

"
