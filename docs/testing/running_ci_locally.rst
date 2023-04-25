Running CI Locally
------------------

This document aims to detail how to install the gitlab-runner and run a CI
script locally.


Instructions on intalling the gitlab runner are here: https://docs.gitlab.com/runner/install/

For ubuntu concise instructions are:

.. code:: bash

    ARCH="$(dpkg --print-architecture)"
    echo "ARCH=$ARCH"

    # Download the runner
    curl -LJO "https://gitlab-runner-downloads.s3.amazonaws.com/latest/deb/gitlab-runner_${ARCH}.deb"

    ### Hash Verification
    curl -LJO "https://gitlab-runner-downloads.s3.amazonaws.com/latest/release.sha256"
    python -c "if 1:
        import pathlib
        import subprocess
        deb_fname = 'gitlab-runner_${ARCH}.deb'
        fpath = pathlib.Path('release.sha256')
        known_hashes = fpath.read_text().split(chr(10))
        name_to_hash = dict([line.split(chr(9))[::-1] for line in known_hashes if line.strip()])
        x = subprocess.check_output(['sha256sum', deb_fname])
        hashval = x.strip().decode('utf8').split(' ')[0].strip()
        wanthash = name_to_hash['deb/' + deb_fname]
        print(hashval)
        print(wanthash)
        assert wanthash == hashval
        "

    ### GPG Verification
    ## See Also: https://about.gitlab.com/blog/2021/06/16/gpg-key-used-to-sign-gitlab-runner-packages-rotated/
    curl -LJO "https://gitlab-runner-downloads.s3.amazonaws.com/latest/release.sha256.asc"
    curl -JLO "https://packages.gitlab.com/runner/gitlab-runner/gpgkey/runner-gitlab-runner-4C80FB51394521E9.pub.gpg"
    gpg --import runner-gitlab-runner-4C80FB51394521E9.pub.gpg
    gpg --verify release.sha256.asc release.sha256

    # Install the runner
    sudo dpkg -i gitlab-runner_${ARCH}.deb


Now that the gitlab runner is installed try executing the linting job:

.. code:: bash

    gitlab-runner exec docker run_linter


Now try the full strict and full loose job

.. code:: bash

    gitlab-runner exec docker test_full_strict/pyenv-linux

    gitlab-runner exec docker test_full_loose/pyenv-linux
