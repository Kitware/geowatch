IMAGE=registry.smartgitlab.com/kitware/watch:0.7.5-6ae5ccba-strict-pyenv3.11.2-20230627T174359-0400-from-a19930d6

DVC_LORES_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_HIRES_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

IMAGE_LORES_DATA_DPATH=/root/data/dvc-repos/smart_data_dvc-ssd
IMAGE_HIRES_DATA_DPATH=/root/data/dvc-repos/smart_data_dvc-hdd
IMAGE_EXPT_DPATH=/root/data/dvc-repos/smart_expt_dvc

# Note: need to be careful to mount directories that symlinks reference
# **exactly**.
EXPT_CACHE_DIR=$(python -m watch.utils.simple_dvc cache_dir "$DVC_EXPT_DPATH")
LORES_CACHE_DIR=$(python -m watch.utils.simple_dvc cache_dir "$DVC_LORES_DATA_DPATH")
HIRES_CACHE_DIR=$(python -m watch.utils.simple_dvc cache_dir "$DVC_HIRES_DATA_DPATH")


# Ensure models exist locally
python -m watch.utils.simple_dvc request \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/depth_pcd/basicModel2.h5 \
    "$DVC_EXPT_DPATH"/models/depth_pcd/model3.h5


_note_='
Theoretically this would let us write new files such that the host filesytem
could read them without chown, but doing this doesnt let us login as root which
is where all the python stuff is installed.

    --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    --mount "type=bind,source=$DVC_LORES_DATA_DPATH,target=$IMAGE_LORES_DATA_DPATH" \
    --mount "type=bind,source=$DVC_HIRES_DATA_DPATH,target=$IMAGE_HIRES_DATA_DPATH" \
    --mount "type=bind,source=$DVC_EXPT_DPATH,target=$IMAGE_EXPT_DPATH" \
    --mount "type=bind,source=$EXPT_CACHE_DIR,target=$EXPT_CACHE_DIR,readonly" \
    --mount "type=bind,source=$LORES_CACHE_DIR,target=$LORES_CACHE_DIR,readonly" \
    --mount "type=bind,source=$HIRES_CACHE_DIR,target=$HIRES_CACHE_DIR,readonly" \

Other docker run options:
    #--memory=60g --cpus=8  --gpus='"device=1"' \
'

docker run \
    --gpus all \
    --shm-size=60g \
    --volume "$DVC_LORES_DATA_DPATH:$IMAGE_LORES_DATA_DPATH" \
    --volume "$DVC_HIRES_DATA_DPATH:$IMAGE_HIRES_DATA_DPATH" \
    --volume "$DVC_EXPT_DPATH:$IMAGE_EXPT_DPATH" \
    --volume "$EXPT_CACHE_DIR:$EXPT_CACHE_DIR:ro" \
    --volume "$LORES_CACHE_DIR:$LORES_CACHE_DIR:ro" \
    --volume "$HIRES_CACHE_DIR:$HIRES_CACHE_DIR:ro" \
    --env "DVC_LORES_DATA_DPATH=$IMAGE_LORES_DATA_DPATH" \
    --env "DVC_HIRES_DATA_DPATH=$IMAGE_HIRES_DATA_DPATH" \
    --env "DVC_EXPT_DPATH=$IMAGE_EXPT_DPATH" \
    -it $IMAGE \
    bash

geowatch schedule --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
        bas_pxl.test_dataset:
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
            - $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.35
            - 0.375
            - 0.4
            - 0.425
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_LORES_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_LORES_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_LORES_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
        sv_crop.enabled: 1
        sv_crop.minimum_size: '256x256@2GSD'
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.5

        sv_dino_boxes.enabled: 1
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims: 256
        sv_dino_boxes.window_overlap: 0.5
        sv_dino_boxes.fixed_resolution: 3GSD

        sv_dino_filter.enabled: 1
        sv_dino_filter.end_min_score: 0.15
        sv_dino_filter.start_max_score: 1.0
        sv_dino_filter.box_score_threshold: 0.01
        sv_dino_filter.box_isect_threshold: 0.1

        sv_depth_score.enabled: 1
        sv_depth_score.model_fpath:
            - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
            #- $DVC_EXPT_DPATH/models/depth_pcd/model3.h5
        sv_depth_filter.threshold:
            - 0.25
            - 0.30
            - 0.35
    submatrices:
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/AE_R001/imgonly-AE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/BR_R002/imgonly-BR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/CH_R001/imgonly-CH_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/NZ_R001/imgonly-NZ_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/PE_R001/imgonly-PE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_LORES_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HIRES_DATA_DPATH/Aligned-Drop7/BR_R004/imgonly-BR_R004.kwcoco.zip
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_sv_sweep" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --queue_name "_namek_sv_sweep" \
    --pipeline=bas_building_and_depth_vali \
    --skip_existing=1 \
    --run=1



#docker run --shm-size=60g -it registry.smartgitlab.com/kitware/watch:0.7.5-4246df07-strict-pyenv3.11.2-20230627T150436-0400-from-176d15dd-metrics-e3d2ed8
#xdoctest watch/cli/run_metrics_framework.py
#FAILED watch/cli/run_metrics_framework.py::main:0
#FAILED watch/tasks/fusion/methods/channelwise_transformer.py::MultimodalTransformer.forward_step:0
#FAILED tests/test_heterogeneous_model.py::test_heterogeneous_with_split_attention_backbone - RuntimeError: DataLoader worker (pid(s) 6466) exited unexpectedly

#pytest watch/tasks/fusion/methods/channelwise_transformer.py
#pytest tests/test_heterogeneous_model.py

# Pull out baseline tables
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

sudo chown -R "$USER":smart "$DVC_EXPT_DPATH"/_namek_sv_sweep

python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_sv_sweep
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_sv_sweep/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    "
    #--rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004"
    #--rois="PE_R001"
