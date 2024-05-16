*******************************
Getting started with PostgreSQL
*******************************

KWCoco provides a postgresql backend, which can have data acess advantages, but
it requires that a postgres service is running on the local machine. As of
2024-05-16 postgresql support in kwcoco is still experimental, and it expects
specific users and permissions are already setup. In the future this will
become more configurable, for now we provide instructions to setup a postgresql
service that should work on debian-based systems.

.. code:: bash

    # Install PostgreSQL
    sudo apt install postgresql postgresql-contrib -y

    # Ensure it is started as a service
    sudo systemctl start postgresql.service
    sudo systemctl status postgresql.service

    # Setup extensions
    # https://dba.stackexchange.com/questions/37351/postgresql-exclude-using-error-data-type-integer-has-no-default-operator-class
    sudo -u postgres psql -c "CREATE EXTENSION btree_gist;"

    # Create roles and users
    sudo -u postgres createuser --superuser --no-password Admin
    sudo -u postgres createuser --role=Admin admin

    # Weird? This seems to need to not run in a user directory?
    # https://dba.stackexchange.com/questions/54242/cannot-start-psql-with-user-postgres-could-not-change-directory-to-home-user
    cd "/"
    sudo -u postgres psql -c "ALTER USER admin WITH PASSWORD 'admin';"
    sudo -u postgres psql -c "ALTER USER admin WITH CREATEDB;"
    sudo -u postgres psql -c "ALTER USER admin WITH LOGIN;"
    sudo -u postgres psql -c "ALTER USER admin WITH SUPERUSER;"
    sudo -u postgres psql -c "ALTER USER admin WITH REPLICATION;"
    sudo -u postgres psql -c "ALTER USER admin WITH BYPASSRLS;"

    sudo -u postgres createuser  --no-password --replication Maintainer
    sudo -u postgres psql -c "ALTER USER Maintainer WITH CREATEDB;"
    sudo -u postgres psql -c "ALTER USER Maintainer WITH SUPERUSER;"
    sudo -u postgres psql -c "ALTER USER Maintainer WITH REPLICATION;"
    sudo -u postgres psql -c "ALTER USER Maintainer WITH BYPASSRLS;"

    # This will be the kwcoco default user
    sudo -u postgres createuser --role=Maintainer kwcoco
    sudo -u postgres psql -c "ALTER USER kwcoco WITH PASSWORD 'kwcoco_pw';"
    sudo -u postgres psql -c "ALTER USER kwcoco WITH CREATEDB;"


Be sure to also install the Python packages

.. code:: bash

    pip install psycopg2-binary sqlalchemy_utils sqlalchemy


Test your setup

.. code:: bash

    sudo -u postgres createdb test_kwcocodb
    python -c "from sqlalchemy import create_engine; create_engine('postgresql+psycopg2://kwcoco:kwcoco_pw@localhost:5432/test_kwcocodb').connect()"
