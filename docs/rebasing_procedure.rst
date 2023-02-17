Rebasing Procedure
------------------

Given that teams work on different branches, while sharing common code that is updated on the main branch, conflicts and breakage may occur. If we consider team A developing the task ``A_task``, which uses function x from the core source code. At some point, function x is updated, which will cause errors when ``A_task`` is merged into the main branch. To avoid that, we encourage rebasing your working branch to the main branch periodically. To do that, follow the following steps:

- commit all changes to your current branch
- change to the main branch, and pull all changes
- go back to your working branch, and run: `git rebase -i main`
- if conflicts exist, a list of conflicts will show which need to be fixed, otherwise the rebase will happen smoothly.

Reference tutorials:

- [Git Branching and Rebasing](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)
- [Git Interactive Rebase](https://thoughtbot.com/blog/git-interactive-rebase-squash-amend-rewriting-history)
- [Git Rebase and Re-Witing History](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase)
- [Git Rebase](https://www.benmarshall.me/git-rebase/)
- [The Ultimate Guide to Git Merge and Git Rebase](https://www.freecodecamp.org/news/the-ultimate-guide-to-git-merge-and-git-rebase/)
