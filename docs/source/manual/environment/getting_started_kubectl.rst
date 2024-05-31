************************
Getting started with AWS
************************


Prerequisites
-------------

This assumes you have aws setup.

* `get started with aws <getting_started_aws.rst>`_.

Install Kubernetes CLI
----------------------

We summarize the `official kubectl install instructions <https://kubernetes.io/docs/tasks/tools/#kubectl>`_ here.
The following is a script that will install the latest version of kubectl.

.. code:: bash

    mkdir -p "$HOME/tmp/kub"
    cd "$HOME/tmp/kub"

    STABLE=$(curl -L -s https://dl.k8s.io/release/stable.txt)
    curl -LO "https://dl.k8s.io/release/$STABLE/bin/linux/amd64/kubectl"
    curl -LO "https://dl.k8s.io/$STABLE/bin/linux/amd64/kubectl.sha256"
    echo "$(<kubectl.sha256)  kubectl" | sha256sum --check

    PREFIX=$HOME/.local
    chmod +x kubectl
    mkdir -p "$PREFIX"/bin
    cp ./kubectl "$PREFIX"/bin/kubectl
    # ensure $PREFIX/bin is in the PATH


Test that the install worked:

.. code:: bash

    # Ensure that this works.
    kubectl version --client --output=yaml



Next Steps
----------

* `get started with smartflow <../smartflow/getting_started_smartflow.rst>`_.
