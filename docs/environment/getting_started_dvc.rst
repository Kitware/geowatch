Minimal DVC
-----------

Data Version Control (DVC) is program that helps maintain datasets. It has
excellent `documentation <https://dvc.org/doc>`_ but, the number of things that
it can do is enormous. The goal of this document is to provide a brief overview
of minimal DVC usage, focusing only on the case of dataset management. 


First we need to install dvc. Either see the `official docs <https://dvc.org/doc/install>`_ or assuming you have a Python environment:

.. code:: bash

   pip install dvc[ssh,s3]

In this tutorial we assume you understand :

* git

* hashing


**DVC builds on top of git**. 

A "DVC repository" can only exist inside of an existing git repository. In
addition to your repositories root `.git` folder initializing dvc in a git repo
(via `dvc init`) will create a special adjacent `.dvc` folder.


**DVC requires a "remote cache"**

DVC allows you to "check in" large files or folders (via ``dvc add <path>``).
However, these files are not stored in git. Instead ``dvc add`` will *hash*
your file, and copy the data into your "local cache", create a special
``<path>.dvc`` file which just contains the hash. The small ``*.dvc`` files that
contain the hash is what you will commit to the git repo. The files themselves
are stored in your local cache. You can "push" these real files to a remove
with ``dvc push -r <remote-name>``.

When you clone a git repo that contains a DVC repo, you are only cloning the
git repo that contains these small ``*.dvc`` files. To obtain the real files
you can ``dvc checkout <path>.dvc``, and it will fetch the data that matches
that hash from the remote cache and add it to your local cache.


Files managed by DVC will be visible in your local repo, but they will
generally be symlinked to your local cache. For example if you clone 
a dvc git repo, and dvc ``checkout "bigfile.json"``, you will notice that 
``"bigfile.json"`` will actually be symlink like: 
``bigfile.json -> .dvc/cache/7b/ab735272e1b6dd4f0027d8fe123424``.


An overview of the 4 locations to be aware of is illustrated:

.. code:: 


     [ REMOTE-GIT-REPO ]

         * Contains the ".dvc" files, which only store the hash corresponding to the real file

     [ REMOTE-DVC-CACHE ]       

         * Contains the real data (stored in a hashed file tree)

     [ LOCAL-GIT-REPO ]       

         * This will contain the ".dvc" files. The raw data files will also
           appear here, but they will generally by symlinked to your cache
           directory.

     [ LOCAL-DVC-CACHE ]       

         * Usually this lives in your <repo>/.dvc/cache folder, but it can be
           configured to live elsewhere.  This just contains a copy of whatever
           data on the remote-dvc-cache that you "pulled" onto your local machine.

These locations are illustrated in the following image from the DVC docs

.. image:: https://miro.medium.com/max/700/1*VIES1isu2zvmlZhJgIefYA.png
   :height: 100px
   :align: left



Installation
------------

DVC is a pure python package. You can simply pip install it. Ensure that you
include ``[ssh]`` to get the ssh dependencies, otherwise you wont be able to
talk to remote servers. It is best practice to do this in a virtual
environment.


.. code:: bash

    pip install dvc[ssh]


Core Commands
-------------

The following links to the official documentation on core commands to use DVC minimally: 

* ``dvc add`` - https://dvc.org/doc/command-reference/add - Track data files or directories with DVC, by creating a corresponding .dvc file.

* ``dvc checkout`` - https://dvc.org/doc/command-reference/checkout - Update DVC-tracked files and directories in the workspace based on current dvc.lock and .dvc files.

* ``dvc push`` - https://dvc.org/doc/command-reference/push - Upload tracked files or directories to remote storage based on the current dvc.yaml and .dvc files.

* ``dvc pull`` - https://dvc.org/doc/command-reference/pull - Download tracked files or directories from remote storage based on the current dvc.yaml and .dvc files, and make them visible in the workspace.

* ``dvc move`` - https://dvc.org/doc/command-reference/move - Rename a file or directory tracked with a .dvc file, and modify the .dvc file to reflect the change. The .dvc file is renamed if the file or directory has the same base name (typical).

* ``dvc remove`` - https://dvc.org/doc/command-reference/remove - Remove stages from dvc.yaml and/or stop tracking files or directories (and optionally delete them).

* ``dvc init`` - https://dvc.org/doc/command-reference/init - Initialize a DVC project in the current working directory.

* ``dvc unprotect`` - https://dvc.org/doc/command-reference/unprotect - Unprotect tracked files or directories (when hardlinks or symlinks have been enabled with dvc config cache.type).

* ``dvc cache`` - https://dvc.org/doc/command-reference/cache - Contains a helper command to set the cache directory location: dir.

* ``dvc config`` - https://dvc.org/doc/command-reference/config - Get or set project-level (or global) DVC configuration options.

* ``dvc remote`` - https://dvc.org/doc/command-reference/remote - A set of commands to set up and manage data remotes: add, default, list, modify, remove, and rename.


Details
-------

Because a checked out DVC file will be a symlink, if you need to modify a file,
you will generally need to run ``dvc unprotect <path>``, which will replace the
symlink with a copy of the real file. Then you can modify it as desired. Once
you are finished you can run ``dvc add <path>``, which will check in the new
hashed file to the cache and modify the ``<path>.dvc`` file, which can then be
checked into git. You must then run ``dvc push <path>.dvc -r <remote>`` to
ensure the new data exists on the remote, otherwise when others check out your
new ``.dvc`` file, that corresponding hashed data won't exist in the remote
cache!.


You can modify where your local cache directory lives. This is very useful for
shared machines that serve as the remote itself. 

.. code:: bash

    dvc cache dir --local /data/shared/dvc-cache/smart_watch_dvc


You can tell DVC about credentials needed to login to a remote server,
otherwise you will be prompted for a password each time.

.. code:: bash

    dvc remote modify --local horologic user $AD_USERNAME 
    dvc remote modify --local horologic url ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc

    dvc remote modify horologic user jon.crall
    dvc remote modify horologic url ssh://horologic.kitware.com/data/dvc-caches/smart_watch_dvc
    dvc remote modify horologic port 22

    dvc config core.check_update False
    
    

Use Cases
---------

Change the name of a directory managed by dvc. Use `dvc move` on the file itself (not the dvc file).


Change the name of a file inside a directory manged by dvc. Use regular `mv` on the file and then `dvc add` the dvc managed directory.
