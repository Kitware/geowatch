"""
SeeAlso:
    ../cli/find_dvc.py

TODO:
    - [ ] This needs a better API. Something like the API for git/dvc remotes.

Example Usage:

    # List default directories (hard coded ones that exist)
    python -m watch.cli.find_dvc --command=list

    python -m watch.cli.find_dvc --command=add --name=test --path=$HOME --hardware=hdd

    # List after adding
    python -m watch.cli.find_dvc list

    # Now get one
    python -m watch.cli.find_dvc

    # Force it to recall "test"
    python -m watch.cli.find_dvc --name=test

    # Remove the test dir
    python -m watch.cli.find_dvc --command=remove --name=test

    # Final list
    python -m watch.cli.find_dvc --command=list

    python -m watch.cli.find_dvc --hardware=ssd
    python -m watch.cli.find_dvc --hardware=hdd

Example Usage:
    # For Drop4
    smartwatch_dvc add --name=smart_data_hdd --path=$HOME/data/dvc-repos/smart_data_dvc --hardware=hdd --priority=100 --tags=phase2_data
    smartwatch_dvc list
    smartwatch_dvc get smart_data_hdd

"""
import scriptconfig as scfg


class FindDVCConfig(scfg.Config):
    """
    Command line helper to find the path to the watch DVC repo
    """
    default = {
        'command': scfg.Value('find', help='can be find, set, add, list, or remove', position=1),

        'name': scfg.Value(None, help='specify a name to query or store or remove', position=2),

        'hardware': scfg.Value(None, help='Specify hdd, ssd, etc..., Setable and getable property'),

        'priority': scfg.Value(None, help='Higher is more likely. Setable and getable property'),

        'tags': scfg.Value(None, help='User note. Setable and queryable property'),

        'path': scfg.Value(None, help='The path to the dvc repo. Setable and queryable property'),

        'verbose': scfg.Value(1, help='verbosity mode'),

        'must_exist': scfg.Value('auto', help='if True, filter to only directories that exist. Defaults to false except on "find", which is True.')
    }

    @staticmethod
    def main(cmdline=True, **kwargs):
        from watch.utils import util_data
        import ubelt as ub

        cli_config = FindDVCConfig(default=kwargs, cmdline=cmdline)
        config = dict(cli_config)

        command = config.pop('command')
        verbose = config.pop('verbose')
        must_exist = config.pop('must_exist')
        if must_exist == 'auto':
            must_exist = command == 'find'

        if verbose > 1:
            print('config = {}'.format(ub.repr2(cli_config, nl=1)))

        registry = util_data.DataRegistry()
        if command == 'list':
            print(registry.list(**config, must_exist=must_exist))
        elif command == 'add':
            registry.add(**config)
        elif command == 'remove':
            registry.remove(name=config['name'])
        elif command == 'set':
            registry.set(**config)
        elif command == 'find':
            dpath = registry.find(**config, must_exist=must_exist)
            print(dpath)
        else:
            raise KeyError(command)
        return 0


_CLI = FindDVCConfig


if __name__ == '__main__':
    """
    CommandLine:
        smartwatch find_dvc
        python ~/code/watch/watch/cli/find_dvc.py
        python -m watch.cli.find_dvc --register
        python -m watch.cli.find_dvc --mode=list


    """
    _CLI.main()
