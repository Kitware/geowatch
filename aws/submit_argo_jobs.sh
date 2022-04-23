#!/bin/bash
# TODO: need to change name to indicate this is training.

cd "$HOME/code/watch/aws"
argo submit "$HOME/code/watch/aws/ta2_train_workflow.yml" --watch

argo submit dvc_check_workflow.yml --watch


watch_recent_argo_logs(){
    # This is not 100% reliable has race conditions
    WORKFLOW_NAME=$(argo list --running | head -n 2 | tail -n 1 | cut -d' ' -f1)
    echo "WORKFLOW_NAME = $WORKFLOW_NAME"
    WORKFLOW_NAME=ta2-train-8slrf
    argo logs "${WORKFLOW_NAME}" --follow
    #argo logs "${WORKFLOW_NAME}" --follow
    #argo delete "dvc-access-check-*"
    #argo delete dvc-access-check-bswmx   dvc-access-check-wd7qk   dvc-access-check-zcggj   dvc-access-check-vhc9k   dvc-access-check-gtdlr   dvc-access-check-mqttg   dvc-access-check-g6vpc   dvc-access-check-hwvzf   dvc-access-check-kmd4k   
}
