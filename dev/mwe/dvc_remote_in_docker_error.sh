#!/bin/bash
# https://github.com/iterative/dvc/issues/9264#issuecomment-1488308658

TMP_DIR=$HOME/temp/docker-remote-mwe

# Fresh start
rm -rf "$TMP_DIR"

mkdir -p "$TMP_DIR"
mkdir -p "$TMP_DIR/my-repo"

cd "$TMP_DIR/my-repo"
git init

dvc init
dvc config core.autostage true

echo "my file" > my_file.txt
dvc add my_file.txt
dvc remote add aws s3://foo-bar-bucket/subbucket

git add .gitignore
git commit -am "a commit"


docker run --volume "$TMP_DIR"/my-repo:/host-repo -it python bash

### Inside docker

git clone /host-repo/.git /my-repo

pip install dvc

cd /my-repo
dvc remote add host /host-repo/.dvc/cache

dvc pull -r host my_file.txt
