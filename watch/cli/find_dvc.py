import scriptconfig as scfg


class FindDVCConfig(scfg.Config):
    """
    Command line helper to find the path to the watch DVC repo
    """
    default = {
        'hardware': scfg.Value(None, help='can specify hdd or sdd if those are registered'),
    }

    @staticmethod
    def main(cmdline=True, **kwargs):
        import watch
        config = FindDVCConfig(default=kwargs, cmdline=cmdline)  # NOQA
        dpath = watch.find_smart_dvc_dpath(hardware=config['hardware'],
                                           on_error='raise')
        print(dpath)
        return 1


_CLI = FindDVCConfig


if __name__ == '__main__':
    """
    CommandLine:
        smartwatch find_dvc
        python ~/code/watch/watch/cli/find_dvc.py
    """
    _CLI.main()
