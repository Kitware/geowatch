make_backup_branch(){
    __doc__="
    This is a helper bash function that makes a backup branch and
    then returns to the main branch. Only works if the current branch
    is in a clean state.
    "
    PREFIX="$1"
    CURRENT_BRANCH=$(git branch --show-current)
    #TIMESTAMP=$(date --iso-8601=seconds)
    TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
    BACKUP_BRANCHNAME="backup/$CURRENT_BRANCH/$HOSTNAME/${PREFIX}${TIMESTAMP}"
    echo "BACKUP_BRANCHNAME = $BACKUP_BRANCHNAME"
    # Make the new backup
    git checkout -b $BACKUP_BRANCHNAME
    # Go back ot the first branch
    git checkout $CURRENT_BRANCH
}


MERGE_BRANCH=master
TOPIC_BRANCH=rutgers
BRANCH=$(git merge-base $TOPIC_BRANCH $MERGE_BRANCH)
echo "BRANCH = $BRANCH"


# This is the branch I used, which should be these same as what you would get from the above
BRANCH=6f2e8779471bd8cb17b7779d61a909101b917ae5
echo "BRANCH = $BRANCH"


gitk $BRANCH rutgers master

git checkout rutgers
make_backup_branch "orig_state_"

# Force state to be clean (based on the last thing pushed to the server)
# I.E. we are using the server as a backup, make a local back if you need to
git reset --hard origin/rutgers
make_backup_branch "origin_rutgers"



# Ensure clean state on rutgers
BRANCH=6f2e8779471bd8cb17b7779d61a909101b917ae5
git checkout rutgers
git reset --hard origin/rutgers
git checkout rutgers

# Checkout a new branch to make changes in 
git checkout -b rutgers_flatbranch_v1

# Change where current branch points, but dont change the filesystem state
git reset $BRANCH

# Add the files that are new to your branch
PREVIOUSLY_TRACKED_FILES=$(git ls-tree -r rutgers --name-only | sort)
UNTRCKED_FILES=$(git status --porcelain | awk '/^\?\?/ { print $2; }' | sort)
NEED_TO_ADD="$(comm -12  <(echo "$UNTRCKED_FILES") <(echo "$PREVIOUSLY_TRACKED_FILES"))"
git add $NEED_TO_ADD

# Add all modified and new files in a single "squashed" commit
git commit -am "megasquash"

# Now try to rebase that one mega commit onto master
git rebase -i origin/master
