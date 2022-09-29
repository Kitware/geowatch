#!/bin/bash

# Show files that changed between branches
git diff 10afdc54746c724b650f2b9dfc5261e6509dbc04..d4f887cd287e793e1080281afbdb2f7a733fe6b4 | grep "^diff"



git-diff-branch(){
    __doc__="
    Args:
        branch1 : branch to take diff from
        *fpaths : paths to all files to diff
    "
    BRANCH_NAME=$1
    shift
    FPATHS=("$@")

    GIT_ROOT=$(git rev-parse --show-toplevel)
    echo "
    BRANCH_NAME = $BRANCH_NAME
    GIT_ROOT=$GIT_ROOT
    "
    TMP_DPATH=$(mktemp -d)
    for FPATH in "${FPATHS[@]}"; do
        REL_PATH=$(realpath --relative-to="$GIT_ROOT" "$FPATH")
        REL_DPATH=$(dirname "$REL_PATH")
        FNAME=$(basename "$REL_PATH")
        TMP_DIRNAME="$TMP_DPATH/$BRANCH_NAME/$REL_DPATH"
        TMP_FPATH="${TMP_DIRNAME}/${FNAME}"
        mkdir -p "$TMP_DIRNAME"
        #git-branch-diff.XXXXXX)
        git show "${BRANCH_NAME}:${REL_PATH}" > "$TMP_FPATH" && colordiff -U 3 "$TMP_FPATH" "${FPATH}"
    done
    rm -rf "$TMP_DPATH"
}

BRANCH1=10afdc54746c724b650f2b9dfc5261e6509dbc04
#BRANCH2=d4f887cd287e793e1080281afbdb2f7a733fe6b4

git-diff-branch "$BRANCH1" watch/tasks/depth/datasets.py watch/tasks/depth/dzyne_img_util.py watch/tasks/depth/predict.py watch/tasks/landcover/datasets.py
