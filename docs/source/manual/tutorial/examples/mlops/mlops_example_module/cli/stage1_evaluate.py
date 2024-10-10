#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
import kwutil
import json
import rich
from rich.markup import escape


class Stage1EvaluateCLI(scfg.DataConfig):
    pred_fpath = scfg.Value(None, help='path to predicted file')
    true_fpath = scfg.Value(None, help='path to truth file')
    out_fpath = scfg.Value(None, help='path to evaluation file')
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
            name='stage1_evaluate',
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=True,
        )
        proc_context.start()

        print('Load file')
        true_text = ub.Path(config.true_fpath).read_text()
        pred_text = ub.Path(config.pred_fpath).read_text()

        true_hashid = ub.hash_data(true_text)
        pred_hashid = ub.hash_data(pred_text)
        true_int = int(true_hashid, 16)
        pred_int = int(pred_hashid, 16)
        hamming_distance = bin(true_int ^ pred_int).count('1')
        size = (len(true_hashid) * 4)
        acc = (size - hamming_distance) / size

        metrics = {
            'accuracy': acc,
            'hamming_distance': hamming_distance,
        }

        # A dummy evaluate computation
        data['result'] = metrics

        obj = proc_context.stop()
        data['info'].append(obj)

        out_fpath = ub.Path(config.out_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(data))
        print(f'wrote to: out_fpath={out_fpath}')

__cli__ = Stage1EvaluateCLI

if __name__ == '__main__':
    __cli__.main()


    r"""

    CommandLine:
        python ~/code/geowatch/docs/source/manual/tutorial/examples/mlops/mlops_example_module/cli/stage1_evaluate.py \
            --true_fpath ~/.bashrc \
            --pred_fpath ~/.bashrc \
            --out_fpath out.json

        python -m mlops_example_module.cli.stage1_predict
    """
