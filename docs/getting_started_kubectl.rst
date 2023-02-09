************************
Getting started with AWS
************************


Prerequisites
-------------

This assumes you have aws setup.

Install Kubernetes CLI
----------------------

This script will install the latest version of kubectl.

.. code:: bash

    mkdir -p "$HOME/tmp/kub"
    cd "$HOME/tmp/kub"

    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
    echo "$(<kubectl.sha256)  kubectl" | sha256sum --check

    PREFIX=$HOME/.local
    chmod +x kubectl
    mkdir -p "$PREFIX"/bin
    cp ./kubectl "$PREFIX"/bin/kubectl
    # ensure $PREFIX/bin is in the PATH

    # Ensure that this works.
    kubectl version --client --output=yaml

Or follow the `official kubectl install instructions <https://kubernetes.io/docs/tasks/tools/#kubectl>`_.
