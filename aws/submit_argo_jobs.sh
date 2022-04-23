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
