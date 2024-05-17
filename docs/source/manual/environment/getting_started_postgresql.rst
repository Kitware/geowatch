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


As a reminder in geowatch the posgresql backend can be used by setting the
``data.sqlview=postgresql`` in the fit configuration.



Other tests and introspection.

TODO: commands to list tables, print contents of a table, and otherwise debug issues with postgres.

.. code:: bash

   sudo -u postgres psql -c "\list"


   # export PGUSER='postgres'
   # export PGHOST='postgres-host-end-point'
   # export PGPORT=5432
   # PGPASSWORD='uber-secret'

   # set as a name from the previous list command
   export PGDATABASE=_2e294_ganns-KR_R002-rawbands.kwcoco.view.v016_1x.postgresql
   export PGDATABASE=_8a9ed_ganns-KR_R001-rawbands.kwcoco.view.v016_1x.postgresql
   export PGDATABASE=_2e294_ganns-KR_R002-rawbands.kwcoco.view.v016_1x.postgresql
   export PGDATABASE=_85d51_ganns-KR_R001-rawbands.kwcoco.view.v016_2x.postgresql
   #sudo -u postgres psql -c "\c $PGDATABASE"

   export TABLENAME=annotations
   sudo -u postgres psql -d $PGDATABASE -t -q -c \
   "
    SELECT COUNT(*)
    FROM $TABLENAME;
   "

   sudo -u postgres psql -d $PGDATABASE -t -q -c \
     "SELECT table_catalog,table_schema,table_name
       FROM information_schema.tables where table_schema='public';"

   sudo -u postgres psql -d $PGDATABASE -t -q -c \
   "select column_name, data_type, character_maximum_length, column_default, is_nullable
   from INFORMATION_SCHEMA.COLUMNS where table_name = '$TABLENAME';"


   sudo -u postgres psql -d $PGDATABASE -t -q -c \
   "select column_name, data_type, character_maximum_length, column_default, is_nullable
   from INFORMATION_SCHEMA.COLUMNS where table_name = '$TABLENAME';"

   sudo -u postgres psql -d $PGDATABASE -t -q -c \
   "\d+ $TABLENAME"

