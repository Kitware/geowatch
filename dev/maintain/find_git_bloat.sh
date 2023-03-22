

# https://stackoverflow.com/questions/13403069/how-to-find-out-which-files-take-up-the-most-space-in-git-repo
#
# Sort all objects in the repo by size
git rev-list --all --objects | awk '{print $1}' | git cat-file --batch-check | sort -k3nr

# Noting that 2986e51827c9d86b651fe1cb6de8ed9c8842b614 is a very big object

# Find which commit adds a specific object
git log --all --find-object=2986e51827c9d86b651fe1cb6de8ed9c8842b614


# This file seems to correspond to
# watch/tasks/super_res/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth


# We can rewrite history of a branch to remove all instances of a specific
# file.  Note: that this will only work if the file has not been merged into
# main yet.  Otherwise the entire repo will need a history rewrite.

env FILTER_BRANCH_SQUELCH_WARNING=1 \
  git filter-branch -f --prune-empty --index-filter '
    git rm -rf --cached --ignore-unmatch -- watch/tasks/super_res/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
  ' main..HEAD
