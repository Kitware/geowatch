Contributing New Models to the WATCH Project
=============================================

This document describes the required steps to share your model and model 
configuration files within the WATCH team. This guide assumes that you
already have a model and the associated files ready to be shared.

Remember: **NEVER COMMIT LARGE FILES DIRECTLY TO GIT!** Git works well for small files. For large files we use DVC.

NOTE: PyTorch models with hyperparameters are typically are saved as ``.pt`` 
files whereas models files only containing weights are typically saved as 
``.ckpt`` files.


NOTE: Check out `docs/environment/getting_started_dvc.rst <https://gitlab.kitware.com/smart/watch/-/blob/main/docs/environment/getting_started_dvc.rst>`_ for more information
regarding how DVC works.


Assumptions:
------------

* The `smart_expt_dvc <https://gitlab.kitware.com/smart/smart_expt_dvc>`_ has been locally cloned.

* You have copied your model and associated files to the ``smart_expt_dvc`` repository.


Steps to sharing your model:
----------------------------

1. Create a new branch for your model. For example, ``<feature_name>_MM_DD_YYYY``. Using ``git switch -c <feature_name>_MM_DD_YYYY`` will create a new branch and switch to it.

2. Track model and associated files using ``dvc add <file>``. This will create a ``<file>.dvc`` file and move the ``<file>`` to the DVC cache.

3. Track the newly created ``<file>.dvc`` file(s) using ``git add <file>.dvc``.

4. Commit the changes using ``git commit -m "<commit messaage>"``. E.g. ``git commit -m "Add <model_name> model and associated files."``.

5. Push the changes to the remote repository using ``git push``. E.g. ``git push -u origin <feature_name>_MM_DD_YYYY``.

6. Create a merge request to merge your branch into the ``main`` branch.


Whats going on behind the scenes:
---------------------------------
When you begin tracking files via DVC (step 2), your local DVC cache has a reference to that file. 
You can ``dvc push <file> -r <remote>`` to upload the file itself to a remote. However, at this point
other people have no way of getting access to the file you just uploaded. You need to provide them
with the small ``<file>.dvc``, which is what they will use to query the DVC remote for that actual file.
With our current default config that will automatically ``git add <file>.dvc`` for you, but if you 
didn't set that up, then you would need to do that. Then you git commit to add the small ``<file>.dvc``
to the git repo, and git push so other people can git pull and get that small ``.dvc`` file. Now that 
other people have the ``.dvc`` file, they have enough information to retreive the actual data. They 
``dvc pull <file>.dvc`` to do this. 
