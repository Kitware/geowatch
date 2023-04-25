Rebasing Procedure
------------------

Given that teams work on different branches, while sharing common code that is updated on the main branch, conflicts and breakage may occur. If we consider team A developing the task ``A_task``, which uses function x from the core source code. At some point, function x is updated, which will cause errors when ``A_task`` is merged into the main branch. To avoid that, we encourage rebasing your working branch to the main branch periodically. To do that, follow the following steps:


Before starting, we assume you are on your development branch. Check this with ``git status``. Given this, then:

First commit all changes to your current branch

.. code:: bash

  git add <files you have changed or added>
  git commit -m "description of changes"

Then checkout the main branch, and pull to update ``main`` to the latest state.

.. code:: bash

  git checkout main
  git pull 


If there is an error pulling the main branch try:


.. code:: bash

  git fetch origin
  git reset --hard origin/main 

Now that main is up to date, go back to your working branch:

.. code:: bash

    git checkout -
    # OR:
    # git checkout <your_branchname>
   
   
And then start the interactive rebase: 

.. code:: bash

   git rebase -i main


This will prompt you with a list of your commits that should be added on top of
main. Typically you can just accept the list that it finds.

If you have only updated code that hasn't been touched by anyone else, then the
reabse should happen smoothly with no issues. 

Otherwise, it may be the case that conflicts exist. If they do use ``git status`` to determine which files
have conflicts and fix them. Edit them and search for ``=====`` to find the sections with conflicts and resolve them.
Then ``git add`` the resolved files and run ``git rebase --continue``. Repeat these steps until the rebase is finished.
If something goes wrong you can always ``git rebase --abort`` to return your branch to its previous state.

If the rebase does work smoothly the last thing to do is update your branch on
the remote. Because the rebase copies your commits, you will need to overwrite
the status of your branch on the remote by doing a force push: 

.. code:: bash

   git push --force


Now your code is rebased and on a branch on the git remote server.


Note: if you have your dev branch checked out across multiple machines / repos, other
repo checkouts will have to force update to the new rebased branch head. This is done via
checkout on your branch and running

.. code:: bash

   git fetch

   git reset --hard origin/<your_branchname>

Ensure you don't have any changes as it will completely reset the state on that
checkout to the state of the remote.


Reference tutorials:

- `Git Branching and Rebasing <https://git-scm.com/book/en/v2/Git-Branching-Rebasing>`_
- `Git Interactive Rebase <https://thoughtbot.com/blog/git-interactive-rebase-squash-amend-rewriting-history>`_
- `Git Rebase and Re Witing History <https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase>`_
- `Git Rebase <https://www.benmarshall.me/git-rebase/>`_
- `The Ultimate Guide to Git Merge and Git Rebase <https://www.freecodecamp.org/news/the-ultimate-guide-to-git-merge-and-git-rebase/>`_
