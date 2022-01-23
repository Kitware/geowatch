# TODO: need to change name to indicate this is training.


cd "$HOME/code/watch/aws"
#argo submit ta2_train_workflow.yml --watch
argo submit dvc_check_workflow.yml --watch
argo submit dvc_access_check.yml --watch



watch_recent_argo_logs(){
    # This is not 100% reliable has race conditions
    WORKFLOW_NAME=$(argo list --running | head -n 2 | tail -n 1 | cut -d' ' -f1)
    echo "WORKFLOW_NAME = $WORKFLOW_NAME"
    argo logs "${WORKFLOW_NAME}" --follow


    argo delete

}
