"""
SeeAlso:
    ../cli/find_dvc.py

TODO:
    - [ ] This needs a better API. Something like the API for git/dvc remotes.

Example Usage:

    # List default directories (hard coded ones that exist)
    python -m watch.cli.find_dvc --mode=list

    python -m watch.cli.find_dvc --mode=add --name=test --path=$HOME --hardware=hdd

    # List after adding
    python -m watch.cli.find_dvc --mode=list

    # Now get one
    python -m watch.cli.find_dvc

    # Force it to recall "test"
    python -m watch.cli.find_dvc --name=test

    # Remove the test dir
    python -m watch.cli.find_dvc --mode=remove --name=test

    # Final list
    python -m watch.cli.find_dvc --mode=list

    python -m watch.cli.find_dvc --hardware=ssd
    python -m watch.cli.find_dvc --hardware=hdd

"""
import scriptconfig as scfg


class FindDVCConfig(scfg.Config):
    """
    Command line helper to find the path to the watch DVC repo
    """
    default = {
        'hardware': scfg.Value(None, help='can specify hdd or sdd if those are registered'),

        'name': scfg.Value(None, help='specify a name to query or store or remove'),

        'path': scfg.Value(None, help='only used in add mode'),

        'mode': scfg.Value('get', help='can be get, add, list, or remove'),

        # 'register': scfg.Value(False, help='if specified, registers this path as a new DVC directory in ~/.config/watch'),
        # 'list': scfg.Value(False, help='if True, lists registered DVC directories'),
    }

    @staticmethod
    def main(cmdline=True, **kwargs):
        from watch.utils import util_data
        config = FindDVCConfig(default=kwargs, cmdline=cmdline)  # NOQA

        if config['mode'] == 'list':
            import pandas as pd
            candiates = util_data._dvc_registry_list()
            candiates = pd.DataFrame(candiates)
            print(candiates.to_string())
        elif config['mode'] == 'add':
            util_data._dvc_registry_add(
                name=config['name'], path=config['path'],
                hardware=config['hardware'])
            import pandas as pd
            candiates = util_data._register_list_smart_dvc_path()
            candiates = pd.DataFrame(candiates)
            print(candiates.to_string())
        elif config['mode'] == 'remove':
            util_data._dvc_registry_remove(name=config['name'])
        elif config['mode'] == 'get':
            dpath = util_data.find_smart_dvc_dpath(
                hardware=config['hardware'],
                name=config['name'],
                on_error='raise',
            )
            print(dpath)
        else:
            raise KeyError(config['mode'])

        return 1


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
