"""
These are the current models that should be considered for use in production.
This also contains metadata about what data the models expect to run on.
(This should also be contained in the model metadata itself).
"""


PRODUCTION_MODELS = [
    {
        'name': 'BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135',
        'task': 'BAS',
        'file_name': 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135.pt',
        'input_channels': 'blue|green|red|nir|swir16|swir22',
        'sensors': ['L8', 'S2', 'WV'],
        'train_dataset': 'Drop1-Aligned-L1-2022-01/combo_DILM_train.kwcoco.json',
    },
    {
        'name': 'BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819',
        'task': 'SC',
        'file_name': 'models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt',
        'input_channels': 'blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
        'sensors': ['L8', 'S2', 'WV'],
        'train_dataset': 'Drop1-Aligned-L1-2022-01/combo_DILM_train.kwcoco.json',
    },

    {
        'name': 'SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679',
        'file_name': 'models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679.pt',
        'task': 'SC',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_c001_v076_epoch=90-step=186367',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=90-step=186367.pt',
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_c001_v082_epoch=42-step=88063',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt',
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv_train.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    },

    {
        'name': 'BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917',
        'file_name': 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt',
        'task': 'BAS',
        'train_dataset': 'Drop2-Aligned-L1-2022-01/combo_L_nowv.kwcoco.json',
        'sensors': ['L8', 'S2'],
        'input_channels': 'blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field',
    }
]


# TODO Investigate v53 epoch 3. It might have a really good recall
