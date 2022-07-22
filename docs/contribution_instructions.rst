How to contribute
-----------------

We follow a `merge requests <https://docs.gitlab.com/ee/user/project/merge_requests/>`_ workflow.

Here is a complete, minimal example of how to add code to this repository, assuming you have followed the instructions above. You should be inside this repo's directory tree on your local machine and have the WATCH environment active.

.. code:: bash

   git checkout -b my_new_branch

   # example commit: change some files
   git commit -am "changed some files"

   # example commit: add a file
   echo "some work" > new_file.py
   git add new_file.py
   git commit -am "added a file"

   # now, integrate other changes that have occurred in this time
   git merge origin/master

   # If you are brave, use `git rebase -i origin/master` instead. It produces a
   # nicer git history, but can be more difficult for people unfamiliar with git.

   # make sure you lint your code!
   python dev/lint.py watch

   # make sure all tests pass (including ones you wrote!)
   python run_tests.py

   # and add your branch to gitlab.kitware.com
   git push --set-upstream origin my_new_branch

   # This will print a URL to make a MR (merge request)
   # Follow the steps on gitlab to submit this. Then it will be reviewed.
   # Tests and the linter will run on the CI, so make sure they work
   # on your local machine to avoid surprise failures.


To get your code merged, create an MR from your branch `here <https://gitlab.kitware.com/smart/watch/-/merge_requests>`_ and @ someone from Kitware to take a look at it. It is a good idea to create a `draft MR <https://docs.gitlab.com/ee/user/project/merge_requests/drafts.html>`_ a bit before you are finished, in order to ask and answer questions about your new feature and make sure it is properly tested.

You can use `markdown <https://docs.gitlab.com/ee/user/markdown.html>`_ to write an informative merge message.
