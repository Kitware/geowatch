# POC bloat script with better python interaction
# See Also:
#     New standalone version, that might go into git-well? ~/local/git_tools/git_find_bloat.py
python -c 'if 1:
    import ubelt as ub
    import xdev

    # Define special characters for bash-embedded python
    lbrace = chr(123)
    rbrace = chr(125)
    squote = chr(39)

    # Sort all objects in the repo by size and show the top 20 along with the
    # object ids they are associated with
    awk_cmd = f"awk {squote}{lbrace}print $1{rbrace}{squote}"
    command = f"git rev-list --all --objects | {awk_cmd} | git cat-file --batch-check | sort -k3nr"
    info = ub.cmd(command, shell=True)

    # Build a Python table with object info
    lines = [x for x in info.stdout.split(chr(10)) if x.strip()]
    rows = []
    for line, _ in zip(lines, range(20)):
        obj_id, type, size = line.split(" ")
        rows.append({
            "obj_id": obj_id,
            "type": type,
            "num_bytes": int(size),
            "size": xdev.byte_str(int(size)),
        })

    # For each object in the table, determine if it is "big", and then
    # determine what commits reference it.
    big_threshold = 5000000  # Five megabytes
    big_file_threshold = 100000  # tenth a megabyte

    big_paths = ub.ddict(dict)
    for row in rows:
        if row["num_bytes"] > big_threshold:
            obj_id = row["obj_id"]
            obj_info = ub.cmd(f"git log --all --find-object={obj_id}")
            for line in obj_info.stdout.split(chr(10)):
                if line.startswith("commit "):
                    commit_id = line.split(" ")[1]
                    # Find all of the files associated with this commit
                    commit_info = ub.cmd(f"git diff-tree --no-commit-id --name-only {commit_id} -r")
                    paths = [p for p in commit_info.stdout.split(chr(10)) if p.strip()]

                    for p in paths:
                        p = ub.Path(p)
                        # Find how big the file was at this point in time.
                        result = ub.cmd(f"git ls-tree -r {commit_id} -- {p}").stdout.strip()
                        if result:
                            num_bytes = int(result.split()[0])
                            if num_bytes > big_file_threshold:
                                prow = big_paths[p]
                                prow["path"] = p
                                prow.setdefault("commit_ids", [])
                                prow.setdefault("obj_ids", [])
                                prow["obj_ids"].append(obj_id)
                                prow["commit_ids"].append(commit_id)

                                prow["num_bytes"] = num_bytes
                                prow["size"] = xdev.byte_str(prow["num_bytes"])
                                prow["exists"] = p.exists()
                                if prow["exists"]:
                                    prow["current_num_bytes"] = p.stat().st_size
                                    prow["current_size"] = xdev.byte_str(prow["current_num_bytes"])
                                else:
                                    prow["current_num_bytes"] = 0
                                    prow["current_size"] = 0
            ...
    big_paths = ub.udict(big_paths)
    big_paths = big_paths.sorted_values(key=lambda x: x.get("num_bytes", -1))
    print(ub.urepr(big_paths))

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df)
'

# https://stackoverflow.com/questions/13403069/how-to-find-out-which-files-take-up-the-most-space-in-git-repo
#
# Sort all objects in the repo by size and show the top 20
git rev-list --all --objects | awk '{print $1}' | git cat-file --batch-check | sort -k3nr | head -n 20

# Noting that 2986e51827c9d86b651fe1cb6de8ed9c8842b614 is a very big object


git log --all --find-object=fcba2687fb86ab911222bb73b6ff0b69fbf24527


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



git log --all --find-object=5f365d826723ab12df839d89aec6f0170242aa72

# List all files in a commit
git diff-tree --no-commit-id --name-only 25646375cc337ba4089878fe21d7627da75f341b -r
