#!/bin/bash
# TODO: need to change name to indicate this is training.

argo submit "$HOME/code/watch/aws/ta2_train_workflow.yml" --watch


argo_follow_recent(){
    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
    argo logs "${WORKFLOW_NAME}" --follow
}
argo_follow_recent



exec_shell_into_pod(){
    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
    #kubectl exec "$WORKFLOW_NAME" -- ls -al /root
    kubectl exec --stdin --tty "$WORKFLOW_NAME" -- /bin/bash
}

check_synced_data(){

    # Check synced data from an argo job
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/

    aws s3 --profile iarpa cp s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/text_logs.log text_logs.log
    aws s3 --profile iarpa sync s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/monitor/ tmp-monitor
}


__notes__(){
    argo logs "${WORKFLOW_NAME}" --follow
    argo delete "dvc-access-check-*"
}
