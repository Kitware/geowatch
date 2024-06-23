Manual Docs
===========


..
    cd ~/code/geowatch/docs/source/manual/
    python -c "if 1:
        import xdev
        import ubelt as ub
        dpath = ub.Path('.').absolute()
        walker = xdev.cli.dirstats.DirectoryWalker(dpath).build()
        for node in sorted(walker.graph.nodes):
            if node.suffix in {'.rst', '.md'} and node.stem != 'manual':
                rel_fpath = node.relative_to(dpath)
                print(rel_fpath.parent / rel_fpath.stem)
    "

.. toctree::
    algorithms/fusion_overview
    algorithms/sensorchan_specs
    algorithms/ta2_deep_dive_info
    baselines/baseline-2023-06-22-bas
    baselines/baseline-2023-06-22-joint_bas_sc
    baselines/baseline-2023-06-22-sc_truth
    baselines/baseline-2023-10-12-full
    baselines/baseline-2024-06-11-bas
    baselines/variation-2023-11-14-full
    beginners/beginners_python
    data/access_dvc_repos
    data/access_internal_phase3_dvc_repos
    data/internal_resources
    data/using_geowatch_dvc
    debugging/debugging_cmdqueue
    development/coding_conventions
    development/coding_environment
    development/coding_tips
    development/contributing_new_models
    development/contribution_instructions
    development/rebasing_procedure
    development/roadmap
    development/ta2_feature_integration
    environment/getting_started_aws
    environment/getting_started_dvc
    environment/getting_started_kubectl
    environment/getting_started_postgresql
    environment/getting_started_ssh_keys
    environment/install_python
    environment/install_python_conda
    environment/install_python_pyenv
    environment/installing_geowatch
    environment/understanding_editable_installs
    environment/windows
    examples/README
    faq/cloudmask
    faq/model_prediction
    faq/stac
    misc/refactoring_datasets_and_models
    misc/structure_proposal
    misc/supporting_projects
    mlops
    onboarding
    overview
    smart/smart_ac_tutorial
    smart/smart_ensemble_tutorial
    smartflow/getting_started_smartflow
    smartflow/smartflow_copying_large_files_to_efs
    smartflow/smartflow_running_the_system
    smartflow/smartflow_training_fusion_models
    testing/running_ci_locally
    testing/testing_practices
    tutorial/README
    watch_cli
