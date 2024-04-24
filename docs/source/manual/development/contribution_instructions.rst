How to Contribute
-----------------

We follow a `merge requests <https://docs.gitlab.com/ee/user/project/merge_requests/>`_ workflow.

Here is a complete, minimal example of how to add code to this repository, assuming you have followed the instructions above. You should be inside this repo's directory tree on your local machine and have the GeoWATCH environment active.

.. code:: bash

   git checkout -b my_new_branch

   # example commit: change some files
   git commit -am "changed some files"

   # example commit: add a file
   echo "some work" > new_file.py
   git add new_file.py
   git commit -am "added a file"

   # now, integrate other changes that have occurred in this time
   git merge origin/main

   # If you are brave, use `git rebase -i origin/main` instead. It produces a
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


To get your code merged, create an MR from your branch `here <https://gitlab.kitware.com/computer-vision/geowatch/-/merge_requests>`_ and @ someone from Kitware to take a look at it. It is a good idea to create a `draft MR <https://docs.gitlab.com/ee/user/project/merge_requests/drafts.html>`_ a bit before you are finished, in order to ask and answer questions about your new feature and make sure it is properly tested.

You can use `markdown <https://docs.gitlab.com/ee/user/markdown.html>`_ to write an informative merge message.


Forking Workflow
----------------

Most external developers will not have permission to push directly to the main
repo. The most effective way to contribute changes is to use a forking
workflow. This means you will create a copy of the repo in your own local
gitlab namespace, make changes directly to that repo, and then submit a merge
request (aka pull request using github terminology) from your fork to the main
repo.

This is done by clicking the `fork <https://gitlab.kitware.com/computer-vision/geowatch/-/forks/new>`_ button on
the main repo page. Which then lets you select which namespace you want to fork
the repo to. You should choose the one that corresponds to your user account.
It will also ask you if you want to change the project name, project slug,
which branches you want to include, and what the visibility level is. These
should all have reasonable defaults.

After modifying these settings click "Fork Project", and you now have a fork of
the project, which you have full control over.

My gitlab username is ``jon.crall``, so my fork is populated as:
``https://gitlab.kitware.com/jon.crall/geowatch``.

On your local machine you want to setup your repo to access either the fork or
the main project. This is done via setting up each endpoint as a
`git remote <https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes>`_.
A git remote is simply a named pointer to a particular URL that stores a copy
of the repo. You may have seen the common remote "origin", which is the default
remote name, based on how you cloned the repo. There are many ways to configure
a remote, but the important thing is that you have:

1. a remote that points to your fork and
2. a remote that points to the main repo

I like to use "origin" as the name of the remote that points to the main repo,
and I like to use the namespace of the repo for forks (e.g. the namespace of my
fork is "jon.crall").

For geowatch this can be setup as follows. First clone the main repo. Typically
I like to clone the main repo, so it defaults as the "origin" remote, but the
following instructions are setup such that it will work regardless of if you
cloned the main repo or your fork.

.. code:: bash

   # NOTE: change the namespace to YOUR fork, this lets the rest of the script
   # be run as-is.
   export NAMESPACE=jon.crall

   # The following 2 lines are only necessary if your clone wasn't pointing to the main repo
   # Remove the existing "origin" remote.
   git remote remove origin
   # Add the new origin
   git remote add origin git@gitlab.kitware.com:computer-vision/geowatch.git

   # Add a remote that points to your fork. Its name is the value of NAMESPACE
   git remote add $NAMESPACE git@gitlab.kitware.com:$NAMESPACE/geowatch.git

Now, to push a branch to your fork, you can use: ``git push $NAMESPACE``, and
to update your checkout from the main repo's main branch you can use
``git pull origin main``.

Note: git is extremely flexibile. You could easilly have your fork be on a
different remote service (e.g. github) by changing the URL, however, to support
MRs forking on gitlab is easier. When you push a branch to your fork, you will
be given stdout that instructs you on how to create an MR. When doing that, you
can change the merge target from your fork, to the original repo's main branch,
and then the MR will show up in the main repo.
