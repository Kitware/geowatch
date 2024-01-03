#!/usr/bin/env python3
"""
Extends torch.utils.collect_env with other relevant information

This was merged into torch upstream via:
    https://github.com/pytorch/pytorch/pull/112993

And could be removed in the future.
"""

import scriptconfig as scfg
import ubelt as ub


class CollectEnvCLI(scfg.DataConfig):
    __command__ = 'collect_env'
    # param1 = scfg.Value(None, help='param1')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.collect_env import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CollectEnvCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        from torch.utils import collect_env
        env_info = collect_env.get_env_info()

        run_lambda = collect_env.run
        patterns = {
            'torch', 'tensorflow', 'tensorboard', 'numpy', 'mypy', 'geowatch',
            'seaborn', 'scriptconfig', 'lightning', 'einops', 'timm', 'scipy',
            'scikit-learn', 'kwimage', 'kwcoco', 'kwutil', 'kwarray',
            'cmd-queue', 'classy-vision', 'ndsampler', 'delayed_image', 'dvc',
            'ubelt', 'shapely', 'pandas', 'rasterio', 'osgeo', 'GDAL',
            'opencv', 'nvidia', 'mmcv', 'mmdet', 'matplotlib', 'mmengine'
            'kornia',
        }

        env_dict = env_info._asdict()

        pip_version, pip_list_output = get_pip_packages(run_lambda, patterns)
        env_dict['pip_packages'] = pip_list_output

        pretty = collect_env.pretty_str(env_info.__class__(**env_dict))
        print(pretty)

        # print('env_dict = {}'.format(ub.urepr(env_dict, nl=1, sv=1)))


def get_pip_packages(run_lambda, patterns=None):
    """Returns `pip list` output. Note: will also find conda-installed pytorch
    and numpy packages."""
    import sys
    if patterns is None:
        patterns = {
            "torch",
            "numpy",
            "mypy",
        }
    # People generally have `pip` as `pip` or `pip3`
    # But here it is incoved as `python -mpip`
    def run_with_pip(pip):
        out = run_and_read_all(run_lambda, "{} list --format=freeze".format(pip))
        return "\n".join(
            line
            for line in out.splitlines()
            if any(
                name in line
                for name in patterns
            )
        )

    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'
    out = run_with_pip(sys.executable + ' -mpip')

    return pip_version, out


def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


__cli__ = CollectEnvCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.cli.collect_env
    """
    main()
