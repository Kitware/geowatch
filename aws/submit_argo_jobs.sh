#!/bin/bash
# TODO: need to change name to indicate this is training.

cd "$HOME/code/watch/aws"
argo submit "$HOME/code/watch/aws/ta2_train_workflow.yml" --watch

argo submit dvc_check_workflow.yml --watch


watch_recent_argo_logs(){
    # This is not 100% reliable has race conditions
    NAME_PREFIX="ta2-train-"
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d' ' -f1)
    echo "WORKFLOW_NAME = $WORKFLOW_NAME"
    argo logs "${WORKFLOW_NAME}" --follow
    #argo logs "${WORKFLOW_NAME}" --follow
    #argo delete "dvc-access-check-*"
    #argo delete dvc-access-check-bswmx   dvc-access-check-wd7qk   dvc-access-check-zcggj   dvc-access-check-vhc9k   dvc-access-check-gtdlr   dvc-access-check-mqttg   dvc-access-check-g6vpc   dvc-access-check-hwvzf   dvc-access-check-kmd4k   

}


check_synced_data(){

    # Check synced data from an argo job
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/

    aws s3 --profile iarpa cp s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/text_logs.log text_logs.log
    aws s3 --profile iarpa sync s3://kitware-smart-watch-data/sync_root/ta2-train-xrppp/unknown-user/ToyDataMSI/runs/ToyFusion_smt_it_stm_p8_v001/lightning_logs/version_0/monitor/ tmp-monitor
}
