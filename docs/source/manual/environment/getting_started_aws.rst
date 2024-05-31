************************
Getting started with AWS
************************

This document provides instructions on installing the AWS CLI tool and setting
up credentials on your local machine.


Installing AWS CLI
------------------

This section provides instructions to install the AWS CLI tool.

For more details or troubleshooting, please refer to the the
`official instructions for installing the AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_.

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

    mkdir -p "$HOME/tmp/setup-aws"
    cd "$HOME/tmp/setup-aws"


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
        tCFBV1MgQ0xJIFRlYW0gPGF3cy1jbGlAYW1hem9uLmNvbT6JAlQEEwEIAD4CGwMF
        CwkIBwIGFQoJCAsCBBYCAwECHgECF4AWIQT7Xbd/1cEYuAURraimMQrMRnJHXAUC
        ZMKcEgUJCSEf3QAKCRCmMQrMRnJHXCilD/4vior9J5tB+icri5WbDudS3ak/ve4q
        XS6ZLm5S8l+CBxy5aLQUlyFhuaaEHDC11fG78OduxatzeHENASYVo3mmKNwrCBza
        NJaeaWKLGQT0MKwBSP5aa3dva8P/4oUP9GsQn0uWoXwNDWfrMbNI8gn+jC/3MigW
        vD3fu6zCOWWLITNv2SJoQlwILmb/uGfha68o4iTBOvcftVRuao6DyqF+CrHX/0j0
        klEDQFMY9M4tsYT7X8NWfI8Vmc89nzpvL9fwda44WwpKIw1FBZP8S0sgDx2xDsxv
        L8kM2GtOiH0cHqFO+V7xtTKZyloliDbJKhu80Kc+YC/TmozD8oeGU2rEFXfLegwS
        zT9N+jB38+dqaP9pRDsi45iGqyA8yavVBabpL0IQ9jU6eIV+kmcjIjcun/Uo8SjJ
        0xQAsm41rxPaKV6vJUn10wVNuhSkKk8mzNOlSZwu7Hua6rdcCaGeB8uJ44AP3QzW
        BNnrjtoN6AlN0D2wFmfE/YL/rHPxU1XwPntubYB/t3rXFL7ENQOOQH0KVXgRCley
        sHMglg46c+nQLRzVTshjDjmtzvh9rcV9RKRoPetEggzCoD89veDA9jPR2Kw6RYkS
        XzYm2fEv16/HRNYt7hJzneFqRIjHW5qAgSs/bcaRWpAU/QQzzJPVKCQNr4y0weyg
        B8HCtGjfod0p1A==
        =gdMc
        -----END PGP PUBLIC KEY BLOCK-----
    " | sed -e 's|^ *||' > aws2.pub
    cat aws2.pub
    gpg --import aws2.pub

    # Download the signature and verify the CLI tool is signed by amazon
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig" -o "awscli-exe-linux-x86_64.zip.sig"

    gpg --verify awscli-exe-linux-x86_64.zip.sig awscli-exe-linux-x86_64.zip


.. .. note: if you have jonc's xpgp tools you can edit the trust too
.. .. python ~/local/scripts/xgpg.py edit_trust "FB5DB77FD5C118B80511ADA8A6310ACC4672475C" "ultimate"


Now that we have verified the integrity, install the aws CLI tool to your local
PATH.


.. code:: bash

    # Unzip the downloaded installer
    unzip -o awscli-exe-linux-x86_64.zip

    # If you want to install somewhere else, change the PREFIX variable
    PREFIX="$HOME/.local"
    mkdir -p "$PREFIX"/bin
    ./aws/install --install-dir "$PREFIX/aws-cli" --bin-dir "$PREFIX/bin" --update

    # Check the version
    "$PREFIX"/bin/aws --version


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

Note for admins: You can generate credentials for yourself or others in the
browser. Login to https://us-east-1.console.aws.amazon.com/iamv2/home

You can query your 12-digit account ID as follows:

.. code:: bash

    export AWS_PROFILE="iarpa"
    aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text

Navigate to Users -> <username> -> Security Credentials


Obtaining Credentials
~~~~~~~~~~~~~~~~~~~~~

To obtain credentials, the current point of contact is
yonatan.gefen@kitware.com (as of 2022-10-06). Please send Yoni an email and CC
matt.leotta@kitware.com and jon.crall@kitware.com to request credentials.

We will then start the process of securely sending you your credentials. If you
have a public GPG key, please send that with your request. We will encrypt your
credentials with your GPG public key, send it to you, and then only you can
decrypt it with your GPG private key.

If you don't have a GPG we will use manual Diffie Hellman handshake. Navigate
to https://cryptotools.net/dhe and generate a private and public key. Send the
public key in your email (don't leave this page, until the process is done). We
will then do the same process on our end, and we will send you our public key.
The next step is we will both paste each other's public keys into the webpage
which will establish a shared secret key. Copy down this shared key, you will
need it later.

On our end, we will take your credentials and encrypt them with this shared
secret. We will send you the encrypted data. Then navigate to
https://cryptotools.net/aes, click "decrypt", paste in both the shared secret
key and then the encrypted message. The plaintext credentials will be generated
in the top box. These are your credentials that we will use in the subsequent
steps.

To summarize, here is an example. Alice wants to send Bob the secret message:
"hello world".

* Alice navigates to https://cryptotools.net/dhe, generates a public key: `bSoNKmm2qF2HLo2tG39gVN4c5xuMnBqX6ES4C0nLdOI=`, and sends it to Bob.

* Bob navigates to https://cryptotools.net/dhe, generates a public key: `UYXjuE9QpXASQM8QQmjImECyvIg4MsOwkS3YrTXXLB0=`, and sends it to Alice.

* Alice enters Bob's secret key into her "Public key" on the right.

* Bob enters Alices's secret key into his "Public key" on the right.

* Both Alice and Bob now see a shared secret: `arnE9PLCOHrvKRLAXsrx+Nc4pyCBZtjCoESjo16Fvi8=` appear, which they can now use for encryption and decryption.

* Alice navigates to https://cryptotools.net/aes, enters the plain text "hello world" and uses the shared secret `arnE9PLCOHrvKRLAXsrx+Nc4pyCBZtjCoESjo16Fvi8=` as the encryption key. This generates the encrypted cyphertext `U2FsdGVkX19sofdkwHQvnur20N8KwDULOxqVPkboYxI=`, which Alice can send to Bob.

* Bob receives the cyphertext from Alice, navigates to https://cryptotools.net/aes, and hits the "Decrypt" button. He enters the cyphertext `U2FsdGVkX19sofdkwHQvnur20N8KwDULOxqVPkboYxI=` into the bottom pane, and also enters the shared secret `arnE9PLCOHrvKRLAXsrx+Nc4pyCBZtjCoESjo16Fvi8=` into the key feild. The decryption happens automatically and the secret message appears in the top plaintext box.


Using Credentials
~~~~~~~~~~~~~~~~~

In the credentials file (``$HOME/.aws/credentials``) append the following text
to create credentials associated with the "iarpa" AWS_PROFILE.

.. code:: ini

    [iarpa]
    aws_access_key_id = <YOUR_ACCESS_KEY>
    aws_secret_access_key = <YOUR_SECRET_KEY>


For the config file (``$HOME/.aws/config``), it is important to specify the
region for the iarpa profile. Set output to either text or json.

.. code:: ini

    [profile iarpa]
    region=us-west-2
    output=json


That completes the install. Verify that it worked by attempting to access bucket containing the DVC cache:


.. code:: bash

    aws --profile iarpa s3 ls s3://kitware-smart-watch-data/dvc/


Note the ``--profile iarpa`` tells aws to authenticate using the "iarpa"
profile in our config/credentials. We could also set an environment variable
``export AWS_PROFILE=iarpa``.


The contents of that folder will be a long list of 2 letter folders and temp
files. This is the hashed file structure that the dvc cache uses. include a
"dvc" directory. Seeing this means that you authenticated sucessfully. Note
that when working with DVC you will not need to use the cache directly, we are
simply checking that you have access to it.

AWS Security
------------

It is important to periodically rotate your AWS credentials.

See detailed **internal** instructions for rotating keys:
`here <https://docs.google.com/document/d/1bW8UM1jR3opJ2qf-OU28Yr3Gyg7chZQ2MH5YQYGBIhs/edit#heading=h.z29n19ypsfef>`_.


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



Next Steps
----------

* `getting started with kubectl <getting_started_kubectl.rst>`_.

* `getting started with dvc <getting_started_dvc.rst>`_.
