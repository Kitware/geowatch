

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



# POC bloat script with better python interaction
python -c 'if 1:

    lbrace = chr(123)
    rbrace = chr(125)
    squote = chr(39)

    import ubelt as ub
    awk_cmd = f"awk {squote}{lbrace}print $1{rbrace}{squote}"
    info = ub.cmd(f"git rev-list --all --objects | {awk_cmd} | git cat-file --batch-check | sort -k3nr", shell=True)

    lines = [x for x in info.stdout.split(chr(10)) if x.strip()]
    rows = []
    import xdev
    for line, _ in zip(lines, range(20)):
        obj_id, type, size = line.split(" ")
        rows.append({
            "obj_id": obj_id,
            "type": type,
            "num_bytes": int(size),
            "size": xdev.byte_str(int(size)),
        })

    big_paths = ub.ddict(dict)
    for row in rows:
        if row["num_bytes"] > 874274:
            obj_id = row["obj_id"]
            obj_info = ub.cmd(f"git log --all --find-object={obj_id}")
            for line in obj_info.stdout.split(chr(10)):
                if line.startswith("commit "):
                    commit_id = line.split(" ")[1]
                    commit_info = ub.cmd(f"git diff-tree --no-commit-id --name-only {commit_id} -r")
                    paths = [p for p in commit_info.stdout.split(chr(10)) if p.strip()]

                    for p in paths:
                        p = ub.Path(p)
                        prow = big_paths[p]
                        prow["path"] = p
                        prow.setdefault("commit_ids", [])
                        prow.setdefault("obj_ids", [])
                        prow["obj_ids"].append(obj_id)
                        prow["commit_ids"].append(commit_id)
                        if p.exists():
                            prow["num_bytes"] = p.stat().st_size
                            prow["size"] = xdev.byte_str(prow["num_bytes"])
            ...
    big_paths = ub.udict(big_paths)
    big_paths = big_paths.sorted_values(key=lambda x: x.get("num_bytes", -1))
    print(ub.urepr(big_paths))

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df)

'



git log --all --find-object=5f365d826723ab12df839d89aec6f0170242aa72

# List all files in a commit
git diff-tree --no-commit-id --name-only 25646375cc337ba4089878fe21d7627da75f341b -r
