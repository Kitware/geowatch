
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



config = {
    'absolute': False,
    'dst': '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/data.kwcoco.json',
    'src': ['/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C000.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-NZ_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-RU_C000.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CN_C001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C003.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CH_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-PE_C001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R005.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R004.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R002.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C011.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C012.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R006.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-PE_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R005.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C002.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BH_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R002.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C002.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-BR_R004.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-LT_R001.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_R007.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-ET_C000.kwcoco.json', '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C010.kwcoco.json'],
}
reading datasets  0/31... rate=0 Hz, eta=?, total=0:00:00reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C000.kwcoco.json'
reading datasets  1/31... rate=42.89 Hz, eta=0:00:00, total=0:00:00reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-NZ_R001.kwcoco.json'
reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-RU_C000.kwcoco.json'
reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CN_C001.kwcoco.json'
reading datasets  4/31... rate=1.29 Hz, eta=0:00:20, total=0:00:03reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-KR_R001.kwcoco.json'
reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-AE_C003.kwcoco.json'
reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-US_C001.kwcoco.json'
reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-CH_R001.kwcoco.json'
reading datasets  8/31... rate=1.75 Hz, eta=0:00:13, total=0:00:04reading fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json'
ERROR ex = Exception('Specified fpath=/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop4-2022-07-18-c10-TA1-S2-L8-ACC/imganns-IN_C000.kwcoco.json does not exist. If you are trying to create a new dataset fist create a CocoDataset without any arguments, and then set the fpath attribute. We may loosen this requirement in the future.')
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
