*******************
Setup your SSH Keys
*******************


Generating SSH Keys
-------------------

If you do not have ssh keys generated, you will need to do that.

We recommend creating an ssh identity (using the `ed25519 backend
<https://en.wikipedia.org/wiki/EdDSA>`_). The following bash code shows how to
do this, ensuring the file has the correct permissions.

.. code:: bash

    REMOTE_USERNAME="$USER"  # set to an identifier
    PRIVATE_KEY_FPATH="$HOME/.ssh/id_${REMOTE_USERNAME}_ed25519"

    if [ -f $PRIVATE_KEY_FPATH ]; then
        echo "Found PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"
    else
        echo "Create PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"

        ssh-keygen -t ed25519 -b 256 -f $PRIVATE_KEY_FPATH -N "" -C ""
        echo $PRIVATE_KEY_FPATH

        # Ensure permissions correct and the new key is registered with the ssh-agent
        chmod 700 ~/.ssh
        chmod 400 ~/.ssh/id_*
        chmod 644 ~/.ssh/id_*.pub
        eval "$(ssh-agent -s)"
        ssh-add $PRIVATE_KEY_FPATH
    fi


Remember to ensure the correct permissions:

.. code:: bash

    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/config
    chmod 600 ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/known_hosts
    chmod 400 ~/.ssh/id_*
    chmod 644 ~/.ssh/id_*.pub


Register SSH Keys with a Gitlab
-------------------------------

For official instructions see: https://docs.gitlab.com/ee/user/ssh.html

For the kitware gitlab, navigate to your user preferences, by clicking your
user icon in the top right and then clicking preferences.

Then on the left menu click SSH keys.

Print out the contents of your public key via:

.. code:: bash

    PUBLIC_KEY="$HOME/.ssh/id_${REMOTE_USERNAME}_ed25519.pub"
    cat $PUBLIC_KEY


Then copy/paste that info the prompt for your "Key" on the website. Adding this
will allow you to clone and interact with repos without being prompted for
credentials each time.


Modify An Existing Clone to Use SSH
-----------------------------------

If you cloned the ``geowatch`` repo with the https protocol, you can modify the
remote URL to ssh instead via:


.. code:: bash

    git remote set-url origin git@gitlab.kitware.com:computer-vision/geowatch.git


Register SSH Keys with a DVC server
-----------------------------------

This is for Kitware employees with access to the VPN only.

This tutorial is also slightly older, and needs an update.

By creating pair of public/private SSH keys, you will be able to access git
repos and remote DVC caches without being prompted for server login
credentials.

We will assume you have these following environment variables. Please populate
with your information.

.. code:: bash

    # This is usually Kitware active-directory username
    REMOTE_USERNAME=<your-username-on-the-remote>

    # This is the remote machine that is hosting the data cache
    REMOTE_URI=the-remote-dvc-server.kitware.com

    # Optional: make a one-word name for the server
    REMOTE_NICKNAME=an alias for the server


For example, my username on ``horologic.kitware.com`` is ``jon.crall``, and I
like to refer to the server as ``horologic``.

.. code:: bash

    REMOTE_USERNAME=jon.crall
    REMOTE_URI=horologic.kitware.com
    REMOTE_NICKNAME=$(echo $REMOTE_URI | cut -d. -f1)


Once you have this information, create an ssh identity (
using the `ed25519 backend <https://en.wikipedia.org/wiki/EdDSA>`_). The
following bash code shows how to do this, ensuring the file has the correct
permissions, and also sending the public key to the remote server you want to
authenticate with:

.. code:: bash

    PRIVATE_KEY_FPATH="$HOME/.ssh/id_${REMOTE_USERNAME}_ed25519"

    if [ -f $PRIVATE_KEY_FPATH ]; then
        echo "Found PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"
    else
        echo "Create PRIVATE_KEY_FPATH = $PRIVATE_KEY_FPATH"

        ssh-keygen -t ed25519 -b 256 -f $PRIVATE_KEY_FPATH -N ""
        echo $PRIVATE_KEY_FPATH

        # Ensure permissions correct and the new key is registered with the ssh-agent
        chmod 700 ~/.ssh
        chmod 400 ~/.ssh/id_*
        chmod 644 ~/.ssh/id_*.pub
        eval "$(ssh-agent -s)"
        ssh-add $PRIVATE_KEY_FPATH

        # -----------------------------------------
        # Step 2: Register SSH Keys with dvc remote
        # -----------------------------------------
        # Run ssh-copy-id to let the remote know about your ssh keys
        # You will have to enter your active-directory password here
        ssh-copy-id -i $PRIVATE_KEY_FPATH $REMOTE_USERNAME@$REMOTE_URI
    fi


Depending on your configuation you may need to explicitly register this key
with this remote on your local machine.  Append the appropriate lines to your
``$HOME/.ssh/config`` file:


.. code::

    Host $REMOTE_NICKNAME $REMOTE_URI
        HostName $REMOTE_URI
        Port 22
        User ${REMOTE_USERNAME}
        identityfile "$HOME/.ssh/id_${REMOTE_USERNAME}_ed25519"


If you defined the above environment variables you should be able to run this
code to ensure it exists programatically:


.. code:: bash

    codeblock(){
        __doc__="
        Helper function for unindenting text
        "
        echo "$1" | python -c "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()).strip('\n'))"
    }

    # If the host is not already registered in your config then add it
    HOST_IN_CONFIG="$(cat $HOME/.ssh/config | grep '^ *HostName *'$REMOTE_URI)"
    if [[ "$HOST_IN_CONFIG" == "" ]]; then
        echo "Adding host do your config"
        codeblock "
            # Programatically added bock
            Host $REMOTE_NICKNAME $REMOTE_URI
                HostName $REMOTE_URI
                Port 22
                User ${REMOTE_USERNAME}
                identityfile "$HOME/.ssh/id_${REMOTE_USERNAME}_ed25519"
        " >> $HOME/.ssh/config
        chmod 600 ~/.ssh/config
    else
        echo "Host was already in your config"
    fi


For the working example variables it may look like this:

.. code::

    Host horologic horologic.kitware.com
        HostName horologic.kitware.com
        Port 22
        User jon.crall
        identityfile ~/.ssh/id_jon.crall_ed25519


Remember to ensure the correct permissions:

.. code:: bash

    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/config
    chmod 600 ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/known_hosts
    chmod 400 ~/.ssh/id_*
    chmod 644 ~/.ssh/id_*.pub



Troubleshooting SSH Keys
------------------------

If you receive a permission error when you do a git pull and you are sure your
public ssh key is correctly registered with gitlab, you can do the following to
force git to use a particular ssh key.


.. code:: bash

    export GIT_SSH_COMMAND="ssh -i <path-to-key>"

    # OR

    git config --local core.sshCommand 'ssh -i <path-to-key>'


Information from `SO41385199 <https://stackoverflow.com/questions/41385199/force-git-to-use-specific-key-pub>`_.
