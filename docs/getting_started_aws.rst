************************
Getting started with AWS
************************

This document provides instructions on installing the AWS CLI tool and setting
up credentials on your local machine.


Installing AWS CLI
------------------

This section provides instructions to install the AWS CLI tool.

For more details or troubleshooting, please refer to the the 
`official instructions for installing the AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>_`.

We attempt to summarize the above with a series of commands that should "just
work" and install the `aws` tool on your machine. This does assume a Linux
Ubuntu machine with an x86_64 processor. For other systems refer to the
official docs.

It is also important to have the curl, zip, and gpg packages installed.

.. code:: bash

    # Ensure you have curl on your system
    dpkg -l curl > /dev/null || sudo apt install curl -y

    # Ensure unzip is installed
    dpkg -l zip > /dev/null || sudo apt install zip -y

    # Ensure gpg is installed
    dpkg -l gnupg > /dev/null || sudo apt install gnupg -y


We recommend running the following in a temporary directory.

.. code:: bash

    mkdir -p $HOME/tmp
    cd $HOME/tmp


The first step is to download the aws cli tool.

.. code:: bash

    # Download the CLI tool for linux
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscli-exe-linux-x86_64.zip"

The next step is to verify the integrity of the downloaded tool. This step can
be skipped, but it is usually a good idea to do this.

.. code:: bash

    # Import the amazon GPG public key
    echo "
        -----BEGIN PGP PUBLIC KEY BLOCK-----

        mQINBF2Cr7UBEADJZHcgusOJl7ENSyumXh85z0TRV0xJorM2B/JL0kHOyigQluUG
        ZMLhENaG0bYatdrKP+3H91lvK050pXwnO/R7fB/FSTouki4ciIx5OuLlnJZIxSzx
        PqGl0mkxImLNbGWoi6Lto0LYxqHN2iQtzlwTVmq9733zd3XfcXrZ3+LblHAgEt5G
        TfNxEKJ8soPLyWmwDH6HWCnjZ/aIQRBTIQ05uVeEoYxSh6wOai7ss/KveoSNBbYz
        gbdzoqI2Y8cgH2nbfgp3DSasaLZEdCSsIsK1u05CinE7k2qZ7KgKAUIcT/cR/grk
        C6VwsnDU0OUCideXcQ8WeHutqvgZH1JgKDbznoIzeQHJD238GEu+eKhRHcz8/jeG
        94zkcgJOz3KbZGYMiTh277Fvj9zzvZsbMBCedV1BTg3TqgvdX4bdkhf5cH+7NtWO
        lrFj6UwAsGukBTAOxC0l/dnSmZhJ7Z1KmEWilro/gOrjtOxqRQutlIqG22TaqoPG
        fYVN+en3Zwbt97kcgZDwqbuykNt64oZWc4XKCa3mprEGC3IbJTBFqglXmZ7l9ywG
        EEUJYOlb2XrSuPWml39beWdKM8kzr1OjnlOm6+lpTRCBfo0wa9F8YZRhHPAkwKkX
        XDeOGpWRj4ohOx0d2GWkyV5xyN14p2tQOCdOODmz80yUTgRpPVQUtOEhXQARAQAB
        tCFBV1MgQ0xJIFRlYW0gPGF3cy1jbGlAYW1hem9uLmNvbT6JAlQEEwEIAD4WIQT7
        Xbd/1cEYuAURraimMQrMRnJHXAUCXYKvtQIbAwUJB4TOAAULCQgHAgYVCgkICwIE
        FgIDAQIeAQIXgAAKCRCmMQrMRnJHXJIXEAChLUIkg80uPUkGjE3jejvQSA1aWuAM
        yzy6fdpdlRUz6M6nmsUhOExjVIvibEJpzK5mhuSZ4lb0vJ2ZUPgCv4zs2nBd7BGJ
        MxKiWgBReGvTdqZ0SzyYH4PYCJSE732x/Fw9hfnh1dMTXNcrQXzwOmmFNNegG0Ox
        au+VnpcR5Kz3smiTrIwZbRudo1ijhCYPQ7t5CMp9kjC6bObvy1hSIg2xNbMAN/Do
        ikebAl36uA6Y/Uczjj3GxZW4ZWeFirMidKbtqvUz2y0UFszobjiBSqZZHCreC34B
        hw9bFNpuWC/0SrXgohdsc6vK50pDGdV5kM2qo9tMQ/izsAwTh/d/GzZv8H4lV9eO
        tEis+EpR497PaxKKh9tJf0N6Q1YLRHof5xePZtOIlS3gfvsH5hXA3HJ9yIxb8T0H
        QYmVr3aIUes20i6meI3fuV36VFupwfrTKaL7VXnsrK2fq5cRvyJLNzXucg0WAjPF
        RrAGLzY7nP1xeg1a0aeP+pdsqjqlPJom8OCWc1+6DWbg0jsC74WoesAqgBItODMB
        rsal1y/q+bPzpsnWjzHV8+1/EtZmSc8ZUGSJOPkfC7hObnfkl18h+1QtKTjZme4d
        H17gsBJr+opwJw/Zio2LMjQBOqlm3K1A4zFTh7wBC7He6KPQea1p2XAMgtvATtNe
        YLZATHZKTJyiqA==
        =vYOk
        -----END PGP PUBLIC KEY BLOCK-----
    " | sed -e 's|^ *||' > aws.pub
    cat aws.pub
    gpg --import aws.pub

    # Download the signature and verify the CLI tool is signed by amazon
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig" -o "awscli-exe-linux-x86_64.zip.sig"

    gpg --verify awscli-exe-linux-x86_64.zip.sig awscli-exe-linux-x86_64.zip 


Now that we have verified the integrity, install the aws CLI tool to your local
PATH.

.. code:: bash

    # Unzip the downloaded installer
    unzip -o awscli-exe-linux-x86_64.zip 

    # If you want to install somewhere else, change the PREFIX variable
    PREFIX="$HOME/.local"
    mkdir -p $PREFIX/bin
    ./aws/install --install-dir $PREFIX/aws-cli --bin-dir $PREFIX/bin --update


Note the value of ``PREFIX`` in the above step. The directory ``$PREFIX/bin``
should be in your PATH. If you do not have that location in your path we
recommend adding it like this:

.. code:: bash
    
    # Add to the path in the current shell
    export PATH=$HOME/.local/bin/:$PATH

    # Add the line to your bashrc so all new shells will have the local bin in
    # your path
    echo 'export PATH=$HOME/.local/bin/:$PATH' >> $HOME/.bashrc


Test that your new AWS CLI is working by running:

.. code:: bash

   aws --version
    

Now that you have the AWS CLI, the next step is to ensure you have the correct
credentials.


AWS Credentials
---------------

This document is designed for **internal collaberators** and will provide
instructions on setting up credentials for an IARPA profile, which will give
you access to the SMART S3 buckets.

To use the AWS CLI (and by extension a DVC AWS remote), you must have
credentials and a config.  The default location to store credentials is:
``$HOME/.aws/credentials`` The default location to store a config is:
``$HOME/.aws/config``.

In the credentials file (``$HOME/.aws/credentials``) append the following text
to create credentials associated with the "iarpa" AWS_PROFILE.

.. code:: config

    [iarpa]
    aws_access_key_id = <YOUR_ACCESS_KEY>
    aws_secret_access_key = <YOUR_SECRET_KEY>


For the config file (``$HOME/.aws/config``), it is important to specify the
region for the iarpa profile. Set output to either text or json.

.. code:: config

    [profile iarpa]
    region=us-west-2
    output=json


That completes the install. Verify that it worked by attempting to access bucket containing the DVC cache:


.. code:: bash

    aws --profile iarpa s3 ls s3://kitware-smart-watch-data/


Note the ``--profile iarpa`` tells aws to authenticate using the "iarpa"
profile in our config/credentials. We could also set an environment variable
``export AWS_PROFILE=iarpa``.


The contents of that folder should include a "dvc" directory. This means that
you authenticated sucessfully.

AWS Security
------------

It is important to periodically rotate your AWS credentials.

See detailed **internal** instructions for rotating keys: 
`here <https://docs.google.com/document/d/1bW8UM1jR3opJ2qf-OU28Yr3Gyg7chZQ2MH5YQYGBIhs/edit#heading=h.z29n19ypsfef>_`.


.. code:: bash

    # Install the AWS key rotation script

    [[ -d $HOME/code/aws-rotate-iam-keys ]] || git clone https://github.com/rhyeal/aws-rotate-iam-keys.git $HOME/code/aws-rotate-iam-keys
    cp $HOME/code/aws-rotate-iam-keys/src/bin/aws-rotate-iam-keys $HOME/.local/bin

    cat $HOME/.aws/config
    cat $HOME/.aws/credentials

    # Execute key rotation on your local machine on the IARPA profile
    export AWS_PROFILE=iarpa
    aws-rotate-iam-keys --profile $AWS_PROFILE

    # Synchronize those keys to all other machine that need them.
    # Doing this will depend on how the user synchronizes keys.

