#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
import kwutil
import rich
import json
from rich.markup import escape


class Stage1PredictCLI(scfg.DataConfig):
    src_fpath = scfg.Value(None, help='path to input file')
    dst_fpath = scfg.Value(None, help='path to output file')
    dst_dpath = scfg.Value(None, help='path to output directory')

    param1 = scfg.Value(None, help='some important parameter')
    workers = scfg.Value(0, help='number of parallel workers')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        data = {
            'info': [],
            'result': None
        }

        proc_context = kwutil.ProcessContext(
            name='stage1_predict',
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=True,
        )
        proc_context.start()

        print('Load file')
        text = ub.Path(config.src_fpath).read_text()

        # A dummy prediction computation
        data['result'] = ub.hash_data(str(config.param1) + str(text))

        obj = proc_context.stop()
        data['info'].append(obj)

        dst_fpath = ub.Path(config.dst_fpath)
        dst_fpath.parent.ensuredir()

        dst_fpath.write_text(json.dumps(data))
        print(f'Wrote to: dst_fpath={dst_fpath}')

__cli__ = Stage1PredictCLI

if __name__ == '__main__':
    __cli__.main()

    r"""

    CommandLine:
        python ~/code/geowatch/docs/source/manual/tutorial/examples/mlops/mlops_example_module/cli/stage1_predict.py \
            --src_fpath ~/.bashrc \
            --dst_fpath ./stage1out/out.json
    """
