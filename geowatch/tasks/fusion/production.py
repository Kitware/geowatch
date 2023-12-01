"""
OLD File, but still has relevant information.

These are the current models that should be considered for use in production.
This also contains metadata about what data the models expect to run on.
(This should also be contained in the model metadata itself).

Production code exists here:
    https://gitlab.kitware.com/smart/watch/-/blob/dev/eval3-integration/scripts/run_bas_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch/geowatch/mlops/smart_global_helper.py
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
        'name': 'Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
        'file_name': 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
        'predictions': 'models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976',
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
    # Phase2 Eval: 2020-08-31
    {
        'name': 'Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt',
        'tags': 'phase2_expt',
        'file_name': 'models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt',
        'task': 'BAS',
    },
    {
        'name': 'Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt',
        'tags': 'phase2_expt',
        'file_name': 'models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt',
        'task': 'SC',
    },
    # Phase2 Eval: 2020-11-21
    {
        'name': 'package_epoch0_step41.pt.pt',
        'tags': 'phase2_expt',
        'file_name': 'models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt',
        'task': 'BAS',
    },
    {
        'name': 'Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt',
        'tags': 'phase2_expt',
        'file_name': 'models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt',
        'task': 'SC',
    }
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

NEW_PRODUCTION_MODELS = [
    """
    {
       "rank": [
          1,
          "2022-10-01T224553-5"
       ],
       "model": "Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt",
       "file_name": "./models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt",
       "pred_params": {
          "tta_fliprot": 0,
          "tta_time": 0,
          "chip_overlap": 0.3,
          "input_space_scale": "15GSD",
          "window_space_scale": "10GSD",
          "output_space_scale": "auto",
          "time_span": "auto",
          "time_sampling": "auto",
          "time_steps": "auto",
          "chip_dims": "auto",
          "set_cover_algo": "None",
          "resample_invalid_frames": 1,
          "use_cloudmask": 1
       },
       "track_params": {
          "thresh": 0.1,
          "morph_kernel": 3,
          "norm_ord": 1,
          "agg_fn": "probs",
          "thresh_hysteresis": "None",
          "moving_window_size": "None",
          "polygon_fn": "heatmaps_to_polys"
       },
       "fit_params": {
          "accelerator": "gpu",
          "accumulate_grad_batches": 4,
          "arch_name": "smt_it_stm_p8",
          "attention_impl": "exact",
          "batch_size": 1,
          "change_head_hidden": 2,
          "change_loss": "cce",
          "channels": "*:BGRN|S|H",
          "chip_dims": [
             380,
             380
          ],
          "chip_overlap": 0.0,
          "class_head_hidden": 2,
          "class_loss": "focal",
          "class_weights": "auto",
          "datamodule": "KWCocoVideoDataModule",
          "decoder": "mlp",
          "decouple_resolution": false,
          "devices": "0,",
          "diff_inputs": false,
          "dist_weights": true,
          "dropout": 0.1,
          "global_change_weight": 0.0,
          "global_class_weight": 0.0,
          "global_saliency_weight": 1.0,
          "gradient_clip_algorithm": "value",
          "gradient_clip_val": 0.5,
          "ignore_dilate": 0,
          "init": "Drop3_SpotCheck_V323_epoch=18-step=12976.pt",
          "learning_rate": 0.0001,
          "match_histograms": false,
          "max_epoch_length": 2048,
          "max_epochs": 160,
          "max_steps": -1,
          "method": "MultimodalTransformer",
          "min_spacetime_weight": 0.5,
          "modulate_class_weights": "",
          "multimodal_reduce": "max",
          "name": "Drop4_BAS_Retrain_V002",
          "neg_to_pos_ratio": 0.25,
          "negative_change_weight": 1.0,
          "normalize_inputs": 1024,
          "normalize_perframe": false,
          "optimizer": "AdamW",
          "patience": 160,
          "positive_change_weight": 1.0,
          "precision": 32,
          "resample_invalid_frames": true,
          "saliency_head_hidden": 2,
          "saliency_loss": "focal",
          "saliency_weights": "auto",
          "set_cover_algo": "approx",
          "space_scale": "30GSD",
          "squash_modes": false,
          "stochastic_weight_avg": false,
          "stream_channels": 16,
          "temporal_dropout": 0.5,
          "time_sampling": "soft2+distribute",
          "time_span": "6m",
          "time_steps": 11,
          "token_norm": "None",
          "tokenizer": "linconv",
          "track_grad_norm": -1,
          "true_multimodal": true,
          "upweight_centers": true,
          "use_centered_positives": true,
          "use_cloudmask": 1,
          "use_conditional_classes": true,
          "use_grid_positives": true,
          "weight_decay": 1e-05,
          "window_size": 8,
          "bad_channels": false,
          "sensorchan": "*:BGRN|S|H"
       },
       "metrics": {
          "coi_mAP": NaN,
          "coi_mAUC": NaN,
          "salient_AP": 0.28347576365492144,
          "salient_AUC": 0.9234889970365587,
          "BAS_F1": 0.6666666667000001,
          "test_dset": "Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_kr1br2.kwcoco"
       }
    }
    """
]
