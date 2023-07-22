Understanding Editable Installs
===============================

This document provides a brief overview of how Python packages interact in your
system.


Within a Python (virtual) environment there are several ways that
`the Python import system <https://docs.python.org/3/reference/import.html>`_
determines which package you mean when you run ``import packagename``.


It needs to resolve ``packagename`` to a path containing the package, and it
does this by searching in
`the following order <https://docs.python.org/3/tutorial/modules.html#the-module-search-path>`_:


1. The current working directory.

2. Each directory listed in the ``PYTHONPATH`` environment variable (which is reflected as ``sys.path`` inside Python interpreter).

3. The "site-packages" directory.

In 99% of cases the final package it finds will be in the site-packages
directory. This is because 1. Quality code should not rely on a user being
in a specific working directory and, 2. The user should never need to modify
their ``PYTHONPATH`` as a manner of routine (modifying ``PYTHONPATH`` is for
rare hacks). Thus for the rest of this document we will focus on the
site-packages case. This begs the question: "Where is my site packages?". The
following code snippet will print it out for you.

.. code:: bash

    python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"


Installing to Site Packages
---------------------------

A Python package can be installed into your site-packages directory in two ways:

1. A normal static installation where the entire package is copied into your site-packages directory.

2. An development editable installation, where a link to your code is copied into site-packages, thus changes are reflected immediately.


Whenever you type ``pip install packagename``, pip will download all of the
code associated with packagename and copy it directly into your your
site-packages folder.


A similar story occurs when you pass pip a path to a package repo with a
``setup.py`` or ``pyproject.toml``. In this case,
``pip install <path-to-repo>`` or if you are currently in that repo
``pip install .`` will *copy* all of the code from your repo into
site-packages.  Thus, if you were to make a modification to your original code,
in general, it would not be reflected because the code in site-packages was an
exact copy of your code at install time.

However, this leads to a GOTCHA. If you are in the repo directory, and you have
previously done a full copy install, and you open IPython and type
``import packagename``, the import mechanism will see your current working path
*before* it sees the copy of the package in site-packages, so it might seem
like changes you are making are reflected, but if you move to a different
directory, you will get the package in site-packages.

To remedy this, we typically use an editable install.
Instead of running ``pip install <path-to-repo>``
we run ``pip install -e <path-to-repo>`` or more
typically in the repo directory:  ``pip install -e .``. This means no matter
where you are in your file system, when you type ``import packagename`` you
will get the one you are developing on.


GOTCHA: Even though editable installs reflect changes immediately, they only
reflect changes up to the point where the module was imported, so if you run a
long running script, but make changes, the code that is executing will not see
those changes. However in IPython you can use the
`autoreload plugin <https://ipython.org/ipython-doc/3/config/extensions/autoreload.html>`_
to make code refresh between statements.

GOTCHA: You can get your system into a weird state where site-packages has both
an editable install and a static install of the same package. The general
strategy to deal with this is to keep uninstalling the package until pip says
that there is nothing left to uninstall. Typically this means you run ``pip
install packagename`` 2 or 3 times and then you can ``pip install -e .`` to get
a clean development install of the package.

GOTCHA: Any console scripts (i.e. the invocations of programs specified in the
``entry_points['console_scripts']`` section of ``setup.py``) are statically
installed even in editable mode. Thus the console scripts change at all you may
need to rerun ``pip install -e .``.


See Also
========

There are several other documents on the topic:

* https://tenthousandmeters.com/blog/python-behind-the-scenes-11-how-the-python-import-system-works/
* https://towardsdatascience.com/understanding-python-imports-init-py-and-pythonpath-once-and-for-all-4c5249ab6355
* https://www.devdungeon.com/content/python-import-syspath-and-pythonpath-tutorial
