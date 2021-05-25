## Ideal Integration Procedure

Given that teams work on different branches, while sharing common code that is updated on the main branch, conflicts and breakage may occur. If we consider team A developing the task A_task, which uses function x from the core source code. At some point, function x is updated, which will cause errors when A_task is merged into the master branch. To avoid that, we encourage rebasing your working branch to the master branch periodically. To do that, follow the following steps:

- commit all changes to your current branch
- change to master branch, and pull all changes
- go back to your working branch, and run: `git rebase -i master`
- if conflicts exist, a list of conflicts will show which need to be fixed, otherwise the rebase will happen smoothly.


