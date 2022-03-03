#!/bin/bash
__doc__="
Minimal example to attempt to reproduce issue where DVC tries to remove data
that should be ignored.

This should be fixed https://github.com/iterative/dvc/issues/4249 But I'm
having an issue with it. This MWE does not currently seem to reproduce it, so
maybe it is something else.
"
BASE_DPATH=$HOME/tmp/dvc_pull_issue


write_dummy_pred_and_evaluation(){
    __doc__="
    Helper to write data similar to our prediction / evaluation script
    "
    MODEL_FPATH=$1
    EXPT_DPATH=$(dirname "$MODEL_FPATH")
    MODEL_NAME=$(basename -s .pt "$MODEL_FPATH")
    PRED_DPATH=$EXPT_DPATH/pred_$MODEL_NAME/dataset
    mkdir -p "$PRED_DPATH/_assets/class1"
    mkdir -p "$PRED_DPATH/_assets/class2"
    mkdir -p "$PRED_DPATH/eval/metrics"
    # Intermediate results that should be ignored by DVC
    head /dev/random > "$PRED_DPATH"/_assets/class1/img1.jpg
    head /dev/random > "$PRED_DPATH"/_assets/class1/img2.jpg
    head /dev/random > "$PRED_DPATH"/_assets/class2/img1.jpg
    head /dev/random > "$PRED_DPATH"/_assets/class2/img2.jpg
    head /dev/random > "$PRED_DPATH"/pred.json
    # The summary should not be ignored by DVC 
    head /dev/random > "$PRED_DPATH"/eval/metrics/summary.json
}


# Create a clean start directory
rm -rf "$BASE_DPATH"
mkdir -p "$BASE_DPATH"
cd "$BASE_DPATH"

# Make a simple repo
mkdir -p "$BASE_DPATH/demo_repo"
cd "$BASE_DPATH/demo_repo"
git init --quiet 
dvc init --quiet
dvc config core.autostage true
dvc config cache.type "symlink,hardlink,copy"
dvc config cache.shared group
dvc config cache.protected true
git config --local receive.denyCurrentBranch "warn"

# This pattern will have local visualizations and raw predictions we do not want to check in
echo "models/*/*/*/_assets" >> .dvcignore
echo "models/*/*/*/pred.json" >> .dvcignore
git add .dvcignore

# Add some data to the repo
mkdir -p "models/expt1"
mkdir -p "models/expt2"
echo "content of model1" > "models/expt1/model1.pt"
echo "content of model2" > "models/expt1/model2.pt"
echo "content of model3" > "models/expt2/model3.pt"
echo "content of model4" > "models/expt2/model4.pt"

dvc add models/expt1 models/expt2
git commit -am "Add data v1"


# Make a clone of the simple repo
cd "$BASE_DPATH"
git clone demo_repo/ demo_repo_clone
cd "$BASE_DPATH/demo_repo_clone"
# Set the remote to the other repo
dvc remote add custom "$BASE_DPATH/demo_repo/.dvc/cache"
dvc pull -r custom --recursive .


# Back to Originl Repo, add in basic eval data
cd "$BASE_DPATH/demo_repo"
write_dummy_pred_and_evaluation models/expt1/model1.pt
write_dummy_pred_and_evaluation models/expt2/model4.pt
dvc add models/expt1 models/expt2
git commit -am "model evals for 1 and 4 from orig repo"


# In the Clone Repo
# Do the same evaluations, but and then pull
cd "$BASE_DPATH/demo_repo_clone"

# Make a file that wouldn't be touched by a pull because it should be in the
# .dvcignore file.
mkdir -p models/expt1/pred_model1/_assets/should-be-ignored
head /dev/random > models/expt1/pred_model1/_assets/should-be-ignored/ignore-me.tmp

git pull
dvc pull -r custom --recursive . 


# Causes:
0% Checkout|                                                                                                                                           |0/6 [00:00<?,     ?file/sfile/directory '/home/joncrall/tmp/dvc_pull_issue/demo_repo_clone/models/expt1/pred_model1/_assets/should-be-ignored/ignore-me.tmp' is going to be removed. Are you sure you want to proceed? [y/n] 















# Re add the experiment directory so the metrics get added.
# The _assets and raw predictions should be ignored
dvc add models/expt1 models/expt2
dvc push -r custom --recursive .
git commit -am "new model evals for 1 and 4"
git push


# Back to original repo 
cd "$BASE_DPATH/demo_repo"
git reset --hard HEAD  # the other repo pushed its state to here, so reset the head
dvc checkout --recursive .  # grab the data made by the other repo


# "Write New Model Evaluations"
write_dummy_pred_and_evaluation models/expt1/model2.pt
write_dummy_pred_and_evaluation models/expt2/model3.pt

# Add new model evaluations again 
dvc add models/expt1 models/expt2
git commit -am "new model evals for 2 and 3"


# Back to the clone
cd "$BASE_DPATH/demo_repo_clone"
git pull
dvc pull -r custom --recursive .


# Back to original repo 
cd "$BASE_DPATH/demo_repo"
# Add some data to the repo
mkdir -p "models/expt3"
echo "content of model5" > "models/expt1/model5.pt"
echo "content of model6" > "models/expt2/model6.pt"
echo "content of model7" > "models/expt3/model7.pt"
write_dummy_pred_and_evaluation models/expt1/model5.pt
write_dummy_pred_and_evaluation models/expt3/model7.pt
dvc add models/expt1 models/expt2 models/expt3
git commit -am "new model evals for 5, 6, and 7"


# Back to the clone
cd "$BASE_DPATH/demo_repo_clone"
git pull

write_dummy_pred_and_evaluation models/expt1/model1.pt

dvc pull -r custom --recursive .

