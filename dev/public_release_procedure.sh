#!/bin/bash
__doc__="
Snapshot the state of the repo for public release
"

PRIVATE_BRANCH=dev/flow51
PUBLIC_BRANCH=initial_public_release


TMP_BRANCH=next_release
git checkout "$PRIVATE_BRANCH"

git checkout -b $TMP_BRANCH
git reset --hard "$PRIVATE_BRANCH"
git reset --soft "$PUBLIC_BRANCH"

rm -rf secrets
rm -rf .gitlab-ci-smart.yml
rm -rf .transcrypt/config

git commit -am "Next public snapshot"

git push public "$TMP_BRANCH":main
