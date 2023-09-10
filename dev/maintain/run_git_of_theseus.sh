pip install git-of-theseus

cd "$HOME"/code/watch
git-of-theseus-analyze "$HOME"/code/watch --procs 4


git-of-theseus-line-plot authors.json --outfile authors-line.png
git-of-theseus-line-plot authors.json --outfile authors-line-norm.png --normalize

git-of-theseus-stack-plot cohorts.json --outfile cohorts-stack.png
git-of-theseus-stack-plot authors.json --outfile authors-stack.png
git-of-theseus-stack-plot authors.json --normalize --outfile authors-stack-norm.png
git-of-theseus-stack-plot exts.json --outfile ext-stack.png

git-of-theseus-survival-plot authors.json  --outfile authors-survival.png
git-of-theseus-survival-plot survival.json --outfile survival-survival.png


cohorts.json
exts.json
dirs.json
domains.json
survival.json

# See Also
# https://github.com/src-d/hercules
