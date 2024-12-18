#!/usr/bin/env bash

# This script is assumed to be run inside the example directory
cd ~/code/geowatch/docs/source/manual/tutorial/examples/mlops

EVAL_DPATH=$PWD/pipeline_output
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'mlops_example_module.pipelines.my_demo_pipeline()'
        matrix:
            stage1_predict.src_fpath:
                - README.rst
                - run_pipeline.sh
            stage1_predict.param1:
                - 123
                - 456
                - 32
                - 33
            stage1_evaluate.workers: 4
    " \
    --root_dpath="${EVAL_DPATH}" \
    --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1


EVAL_DPATH=$PWD/pipeline_output
python -m geowatch.mlops.aggregate \
    --pipeline='mlops_example_module.pipelines.my_demo_pipeline()' \
    --target "
        - $EVAL_DPATH
    " \
    --output_dpath="$EVAL_DPATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - stage1_evaluate
    " \
    --stdout_report="
        top_k: 100
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 1
        show_csv: 0
    " \
    --plot_params="
        enabled: 1
    "

