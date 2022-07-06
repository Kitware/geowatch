"""
These are the current models that should be considered for use in production.
This also contains metadata about what data the models expect to run on.
(This should also be contained in the model metadata itself).

Production code exists here:
    https://gitlab.kitware.com/smart/watch/-/blob/dev/eval3-integration/scripts/run_bas_fusion_eval3_for_baseline.py
"""


PRODUCTION_MODELS = [
    {
        'name': 'BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135',
        'gsd': 10.0,
        'task': 'BAS',
        'file_name': 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135.pt',
        'input_channels': 'blue|green|red|nir|swir16|swir22',
        'sensors': ['L8', 'S2', 'WV'],
        'train_dataset': 'Drop1-Aligned-L1-2022-01/combo_DILM_train.kwcoco.json',
    },
    {
        'name': 'BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819',
        'gsd': 10.0,
        'task': 'SC',
        'file_name': 'models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt',
        'input_channels': 'blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
        'sensors': ['L8', 'S2', 'WV'],
        'train_dataset': 'Drop1-Aligned-L1-2022-01/combo_DILM_train.kwcoco.json',
    },

    {
        'name': 'SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679',
        'file_name': 'models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679.pt',
        'gsd': 10.0,
        'task': 'SC',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_c001_v076_epoch=90-step=186367',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=90-step=186367.pt',
        'gsd': 10.0,
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_c001_v082_epoch=42-step=88063',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt',
        'gsd': 10.0,
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt',
        'gsd': 10.0,
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'Drop3_SpotCheck_V323_epoch=18-step=12976',
        'file_name': 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
        'gsd': 10.0,
        'task': 'BAS',
        'input_channels': 'blue|green|red|nir|swir16|swir22',
        'train_dataset': 'Aligned-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
    },

    {
        'name': 'Drop3_SpotCheck_V323_epoch=18-step=12976.pt ',
        'file_name': 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
        'predictions': 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
        'task': 'BAS',
        'gsd': 10.0,
        'input_channels': 'blue|green|red|nir|swir16|swir22',
        'train_dataset': 'Aligned-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        # TODO: populate this with summary measures so we can get a gist of the model "quality" from this list
        'measures': {
            'salient_AP': 0.27492,
            'BAS_F1': 0.34782,
        }
    },
    {
        'name': 'CropDrop3_SC_V006_epoch=71-step=18431.pt',
        'file_name': 'models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt',
        'task': 'SC',
        'gsd': 1.0,
        'input_channels': 'red|green|blue',
        'train_dataset': 'Cropped-Drop3-TA1-2022-03-10/data_s2_wv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'measures': {
            'coi_mAP': 0.336,
            'mean_F1': 0.4489,
        }
    },
    {
        'name': 'CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt',
        'file_name': 'models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt',
        'task': 'SC',
        'gsd': 1.0,
        'input_channels': '(S2,WV):red|green|blue,S2:invariants:16',
        'train_dataset': 'Cropped-Drop3-TA1-2022-03-10/data_s2_wv_train.kwcoco.json',
        'sensors': ['WV', 'S2'],
        'measures': {
            'coi_mAP': 0.41,
            # 'mean_F1': 0.4489,
            'mean_F1': '0.42477983495000005',
        },
        # TODO: programatically set these, add aliases in case the config is
        # extended, so we remember old hashes
        'pred_cfgstr': 'predcfg_4d9147b0',
        'act_cfgstr': 'actcfg_f1456a39',
        'pred_cfg': {
            'tta_time': 1,
            'tta_fliprot': 0,
            'chip_overlap': 0.3,
        },
        'act_cfg': {
            'boundaries_as': "polys",
            'use_viterbi': "v1,v6",
            'thresh': 0.001,
        }
    },


]


# TODO Investigate v53 epoch 3. It might have a really good recall


# These are good models to consider for BAS
CANDIDATE_BAS_MODELS = [
    'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
    'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V313/Drop3_SpotCheck_V313_epoch=34-step=71679.pt'
    'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V319/Drop3_SpotCheck_V319_epoch=60-step=124927.pt'
    'models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=4-step=26149-v3.pt'
    'models/fusion/eval3_candidates/packages/Drop3_bells_seg_V306/Drop3_bells_seg_V306_epoch=28-step=14847-v1.pt',
]

__notes__ = r'''


DVC_DPATH=$(smartwatch_dvc)
cd $DVC_DPATH

joinby(){
    # https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
    local d=${1-} f=${2-}
    if shift 2; then
      printf %s "$f" "${@/#/$d}"
    fi
}

# Define the candidate models
CANDIDATE_MODELS=(
    "models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt"
    "models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V313/Drop3_SpotCheck_V313_epoch=34-step=71679.pt"
    "models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V319/Drop3_SpotCheck_V319_epoch=60-step=124927.pt"
    "models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=4-step=26149-v3.pt"
    "models/fusion/eval3_candidates/packages/Drop3_bells_seg_V306/Drop3_bells_seg_V306_epoch=28-step=14847-v1.pt"
)
printf "$(joinby "\n" "${CANDIDATE_MODELS[@]}")" > bas-models-of-interest.txt

# Pull models onto the system
dvc pull -r aws $(joinby " " "${CANDIDATE_MODELS[@]}")

# Define the dataset to predict models on
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json

python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        --gpus="$TMUX_GPUS" \
        --model_globstr=bas-models-of-interest.txt \
        --test_dataset="$VALI_FPATH" \
        --enable_pred=1 \
        --enable_eval=0 \
        --chip_overlap=0.3 \
        --skip_existing=0 --backend=tmux --run=0


CANDIDATE_MEASURES=(
    "models/fusion/eval3_candidates/eval/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976*/*/*/eval/curves/measures2.json"
    "models/fusion/eval3_candidates/eval/Drop3_SpotCheck_V313/pred_Drop3_SpotCheck_V313_epoch=34-step=71679*/*/*/eval/curves/measures2.json"
    "models/fusion/eval3_candidates/eval/Drop3_SpotCheck_V319/pred_Drop3_SpotCheck_V319_epoch=60-step=124927*/*/*/eval/curves/measures2.json"
    "models/fusion/eval3_candidates/eval/BASELINE_EXPERIMENT_V001/pred_BASELINE_EXPERIMENT_V001_epoch=4-step=26149-v3*/*/*/eval/curves/measures2.json"
    "models/fusion/eval3_candidates/eval/Drop3_bells_seg_V306/pred_Drop3_bells_seg_V306_epoch=28-step=14847-v1*/*/*/eval/curves/measures2.json"
)
# Pull existing evaluation measures from DVC
dvc pull -r aws $(joinby " " "${CANDIDATE_MEASURES[@]}")

# Run the aggregate script on these models
MEASURE_GLOBSTR=$(joinby "," "${CANDIDATE_MEASURES[@]}")
python -m watch.tasks.fusion.aggregate_results \
    --measure_globstr="$MEASURE_GLOBSTR" \
    --out_dpath="$DVC_DPATH/agg_results/custom" \
    --dset_group_key="*Drop3*combo_LM_nowv_vali*" \
    --classes_of_interest "Site Preparation" "Active Construction" \
    --io_workers=10 --show=True

'''
