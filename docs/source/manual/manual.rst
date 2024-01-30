Manual Docs
===========

.. .. python -c "
.. .. import xdev
.. .. import ubelt as ub
.. .. dpath = ub.Path('.').absolute()
.. .. walker = xdev.cli.dirstats.DirectoryWalker(dpath).build()

.. .. for node in walker.graph.nodes:
.. ..    if node.suffix in {'.rst', '.md'}:
.. ..        rel_fpath = node.relative_to(dpath)
.. ..        print(rel_fpath.parent / rel_fpath.stem)
.. .. "

.. toctree::
    testing/testing_practices
    testing/running_ci_locally
    development/coding_tips
    development/contribution_instructions
    development/rebasing_procedure
    development/ta2_feature_integration
    development/contributing_new_models
    development/roadmap
    development/coding_conventions
    debugging/debugging_cmdqueue
    misc/structure_proposal
    misc/supporting_projects
    environment/installing_geowatch
    environment/install_python_conda
    environment/getting_started_kubectl
    environment/getting_started_ssh_keys
    environment/install_python_pyenv
    environment/install_python
    environment/understanding_editable_installs
    environment/getting_started_aws
    environment/getting_started_dvc
    environment/windows
    baselines/baseline-2023-10-12-full
    baselines/baseline-2023-06-22-sc_truth
    baselines/baseline-2023-06-22-bas
    baselines/variation-2023-11-14-full
    baselines/baseline-2023-06-22-joint_bas_sc
    tutorial/README
    smart/smart_ensemble_tutorial
    smart/smart_ac_tutorial
    algorithms/sensorchan_specs
    algorithms/ta2_deep_dive_info
    algorithms/fusion_overview
    faq/stac
    faq/model_prediction
    data/using_geowatch_dvc
    data/internal_resources
    data/access_dvc_repos
    smartflow/smartflow_copying_large_files_to_efs
    smartflow/smartflow_training_fusion_models
    smartflow/getting_started_smartflow
    smartflow/smartflow_running_the_system
    watch_cli
    onboarding
    overview
    mlops
