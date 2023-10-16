pip install git-of-theseus

cd "$HOME"/code/watch
#git-of-theseus-analyze "$HOME"/code/watch --procs 4
#git-of-theseus-analyze . --procs 4

#pint "1day" "seconds"

git-of-theseus-analyze . --interval "86400" --procs 4


git-of-theseus-line-plot authors.json --outfile authors-line.png
git-of-theseus-line-plot authors.json --outfile authors-line-norm.png --normalize

git-of-theseus-stack-plot cohorts.json --outfile cohorts-stack.png
git-of-theseus-stack-plot authors.json --outfile authors-stack.png
git-of-theseus-stack-plot authors.json --normalize --outfile authors-stack-norm.png
git-of-theseus-stack-plot exts.json --outfile ext-stack.png

git-of-theseus-survival-plot survival.json --outfile survival-survival.png


# See Also
# https://github.com/src-d/hercules
install_hercules(){
    curl -L https://github.com/src-d/hercules/releases/download/v10.7.2/hercules.linux_amd64.gz -O
    7z x hercules.linux_amd64.gz
    chmod +x hercules
    mv hercules "$HOME"/.local/bin/hercules
    rm hercules*

    pip install labours --no-deps
}

run_hercules(){
    hercules .
    hercules --burndown .git
    labours -m burndown-project

    hercules --burndown . > burndown_output
    cat burndown_output | labours -m burndown-project -o git_git.png

    docker run --rm srcd/hercules hercules --burndown --pb https://github.com/git/git | docker run --rm -i -v "$(pwd):/io" srcd/hercules labours -f pb -m burndown-project -o /io/git_git.png
    docker run -v "$(pwd):/io" --rm srcd/hercules hercules --burndown /io/.git | docker run --rm -i -v "$(pwd):/io" srcd/hercules labours -f pb -m burndown-project -o /io/git_git.png
}
