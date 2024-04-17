Coding Environment
******************

This document aims to provide tips to make your development environment easier.


Enable Argcomplete
==================

Like all scriptconfig powered CLIs, the geowatch CLI uses `argcomplete <https://pypi.org/project/argcomplete/>`_.

For BASH environments, to enable the tab-complete feature run:

.. code-block:: bash

    pip install argcomplete
    mkdir -p ~/.bash_completion.d
    activate-global-python-argcomplete --dest ~/.bash_completion.d
    source ~/.bash_completion.d/python-argcomplete

Then to gain the feature in new shells put the following lines in your .bashrc:

.. code-block:: bash

    if [ -f "$HOME/.bash_completion.d/python-argcomplete" ]; then
        source ~/.bash_completion.d/python-argcomplete
    fi

If you know of a way to have this feature "install itself" or avoid requiring
this manual step, please submit an MR!

For non-BASH environemts like Zsh refer to the
`argcomplete documentation <https://kislyuk.github.io/argcomplete/#activating-global-completion>`_.


Useful Aliases
==============

The following aliases have proven useful in circumstances where you have multiple DVC repos you need to quickly switch between:


.. code:: bash

    alias wad='cd $(geowatch_dvc --tags="phase3_data" --hardware="auto")'
    alias wae='cd $(geowatch_dvc --tags="phase3_expt" --hardware="auto")'

    # And if you have HDD / SSD variants of a dataset.
    alias wadh='cd $(geowatch_dvc --tags="phase3_data" --hardware="hdd")'
    alias wads='cd $(geowatch_dvc --tags="phase3_data" --hardware="ssd")'
