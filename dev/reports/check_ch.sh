DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.cli.prepare_teamfeats \
    --base_fpath="$DVC_DATA_DPATH/Drop6/imganns-CH_R001.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=1 \
    --with_cold=0 \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres=0,1 --workers=4 --backend=tmux --run=1

python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6/combo_imganns-CH_R001_I.kwcoco.json
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.thresh:
            - 0.17
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.resolution:
            - null
        bas_poly.moving_window_size:
            - null
            - 100
        bas_poly.min_area_sqkm:
            - 0.0072
        bas_poly.max_area_sqkm:
            - 8
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/drop6/annotations/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/drop6/annotations/region_models
        bas_pxl.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_check_ch" \
    --devices="0,1" --queue_size=8 --print_commands=0 \
    --backend=tmux --queue_name "_check_ch" \
    --pipeline=bas --skip_existing=1 \
    --run=1


#python -m watch.cli.run_tracker \
#    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/pred.kwcoco.zip \
#    --default_track_fn saliency_heatmaps \
#    --track_kwargs '{"agg_fn": "probs", "thresh": 0.17, "polygon_simplify_tolerance": 1, "resolution": "auto", "moving_window_size": 100, "min_area_sqkm": 0.0072, "max_area_sqkm": 8}' \
#    --clear_annots \
#    --site_summary None \
#    --out_site_summaries_fpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/site_summaries_manifest.json \
#    --out_site_summaries_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/site_summaries \
#    --out_sites_fpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/sites_manifest.json \
#    --out_sites_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/sites \
#    --out_kwcoco /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/poly.kwcoco.zip
