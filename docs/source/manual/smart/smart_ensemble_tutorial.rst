SMART Ensemble Tutorial
=======================

TODO: finish me.

Note: if you are following this tutorial, please help me make it better as you
play with it and learn from it!

Ensembles are driven by the watch/cli/coco_average_features.py script, which can be accessed via

.. code::

   geowatch average_features --help


While this tutorial is still being written you can check out doctests inside the CLI file for more examples.


It takes multiple kwcoco files with either saliency or class predictions and
then writes them to a new kwcoco file where those channels are averaged
together. This new kwcoco file can be sent to the tracker for polygon
extraction as if it was a kwcoco file from a single model. The rests of the
pipeline can run from these polygons as-is.


Currently running this in mlops is not possible because there is an assumption
that only one model is used at a time. A new specialized ensemble pipeline will
likely need to be defined to use this effectively, but what you can do is use
mlops to generate the commands required for a single model evaluation and then
manually add in this step to produce the averaged output. Then just plug that
into the tracker inputs and the rest of it should be straight forward to execute.


For example say you have predicted saliency with two models so you have:

* ``model1/pred.kwcoco.json`` and
* ``model2/pred.kwcoco.json``

You could ensemble them like:

.. code:: bash


   geowatch average_features \
       --kwcoco_file_paths \
           model1/pred.kwcoco.zip \
           model2/pred.kwcoco.zip \
       --output_kwcoco_path model_ensemble/averaged.kwcoco.zip \
       --channel_name saliency \
       --sensors all


Then for reference you could grab a template for bas commands like:

.. code::

    geowatch schedule --params="
        matrix:
            bas_pxl.package_fpath:
                - MY_PACKAGE.pt
            bas_pxl.test_dataset:
                - MY_DATASET.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: soft4
            bas_poly.thresh:
                - 0.4
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
            bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
            bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 0
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_poly_viz.enabled: 0
        " \
        --root_dpath="$DVC_EXPT_DPATH/_reference" \
        --backend=serial --queue_name "_reference" \
        --pipeline=bas \
        --skip_existing=0 \
        --run=0


And using this as a reference you might construct a set of command that look like this:


.. code:: bash

    # I copied the pixel predict step twice  and put in some custom paths
    python -m geowatch.tasks.fusion.predict \
        --package_fpath=MY_MODEL1.pt \
        --test_dataset=MY_DATASET.kwcoco.zip \
        --pred_dataset=./_reference/pred/flat/bas_pxl/model1_preds/pred.kwcoco.zip \
        --chip_overlap=0.3 \
        --chip_dims=auto \
        --time_span=auto \
        --time_sampling=soft4 \
        --drop_unused_frames=True  \
        --num_workers=2 \
        --devices=0, \
        --batch_size=1 \
        --with_saliency=True \
        --with_class=False \
        --with_change=False

    python -m geowatch.tasks.fusion.predict \
        --package_fpath=MY_MODEL2.pt \
        --test_dataset=MY_DATASET.kwcoco.zip \
        --pred_dataset=./_reference/pred/flat/bas_pxl/model2_preds/pred.kwcoco.zip \
        --chip_overlap=0.3 \
        --chip_dims=auto \
        --time_span=auto \
        --time_sampling=soft4 \
        --drop_unused_frames=True  \
        --num_workers=2 \
        --devices=0, \
        --batch_size=1 \
        --with_saliency=True \
        --with_class=False \
        --with_change=False

    # Inserting the custom average feature script here.

    geowatch average_features \
       --kwcoco_file_paths \
           ./_reference/pred/flat/bas_pxl/model1_preds/pred.kwcoco.zip \
           ./_reference/pred/flat/bas_pxl/model2_preds/pred.kwcoco.zip \
       --output_kwcoco_path "./_reference/pred/flat/bas_ensemble/bas_ensemble_custom/pred.kwcoco.zip" \
       --channel_name saliency \
       --sensors all

    # The rest of the tracking + eval part of the pipeline is unchanged.

    python -m geowatch.cli.run_tracker \
        --in_file "./_reference/pred/flat/bas_ensemble/bas_ensemble_custom/pred.kwcoco.zip" \
        --default_track_fn saliency_heatmaps \
        --track_kwargs '{"agg_fn": "probs", "thresh": 0.4, "inner_window_size": "1y", "inner_agg_fn": "mean", "norm_ord": "inf", "polygon_simplify_tolerance": 1, "time_thresh": 0.8, "resolution": "10GSD", "moving_window_size": null, "poly_merge_method": "v2", "min_area_square_meters": 7200, "max_area_square_meters": 8000000}' \
        --clear_annots=True \
        --site_summary 'None' \
        --boundary_region './annotations/drop6/region_models' \
        --out_site_summaries_fpath "./_reference/pred/flat/bas_poly/bas_poly_id_custom/site_summaries_manifest.json" \
        --out_site_summaries_dir "./_reference/pred/flat/bas_poly/bas_poly_id_custom/site_summaries" \
        --out_sites_fpath "./_reference/pred/flat/bas_poly/bas_poly_id_custom/sites_manifest.json" \
        --out_sites_dir "./_reference/pred/flat/bas_poly/bas_poly_id_custom/sites" \
        --out_kwcoco "./_reference/pred/flat/bas_poly/bas_poly_id_custom/poly.kwcoco.zip"
    #
    python -m geowatch.cli.run_metrics_framework \
        --merge=True \
        --name "some-name" \
        --true_site_dpath "./annotations/drop6/site_models" \
        --true_region_dpath "./annotations/drop6/region_models" \
        --pred_sites "./_reference/pred/flat/bas_poly/bas_poly_id_custom/sites_manifest.json" \
        --tmp_dir "./_reference/eval/flat/bas_poly_eval/bas_poly_eval_id_custom/tmp" \
        --out_dir "./_reference/eval/flat/bas_poly_eval/bas_poly_eval_id_custom" \
        --merge_fpath "./_reference/eval/flat/bas_poly_eval/bas_poly_eval_id_custom/poly_eval.json"


Note: you could do a similar thing with the more complex ``bas_building_and_depth_vali`` pipeline.

Note: I do plan to eventually support ensembles in mlops, but the above should
work in the meantime, and showing positive results would make me prioritize it
higher.
