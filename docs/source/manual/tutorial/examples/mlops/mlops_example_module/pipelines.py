"""
This file defines a ProcessNode for each CLI process in a pipeline.
It is important that each CLI have well defined input and output paths.
"""
from geowatch.mlops.pipeline_nodes import ProcessNode
from geowatch.mlops.pipeline_nodes import PipelineDAG
import ubelt as ub

# Normally we want to invoke installed Python modules so we can abstract away
# hard coded paths, but for this example we will avoid that for simplicity.
try:
    EXAMPLE_DPATH = ub.Path(__file__).parent
except NameError:
    # for developer convinience
    EXAMPLE_DPATH = ub.Path('~/code/geowatch/docs/source/manual/tutorial/examples/mlops/mlops_example_module').expanduser()


class Stage1_Predict(ProcessNode):
    """
    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/docs/source/manual/tutorial/examples/mlops'))
        >>> from mlops_example_module.pipelines import *  # NOQA
        >>> self = Stage1_Predict()
        >>> print(self.command)
    """
    name = 'stage1_predict'
    executable = f'python {EXAMPLE_DPATH}/cli/stage1_predict.py'

    in_paths = {
        'src_fpath',
    }
    out_paths = {
        'dst_fpath': 'stage1_prediction.json',
        'dst_dpath': '.',
    }
    primary_out_key = 'dst_fpath'

    algo_params = {
        'param1': 1,
    }
    perf_params = {
        'workers': 0,
    }

    def load_result(self, node_dpath):
        import json
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        output_fpath = node_dpath / self.out_paths[self.primary_out_key]
        result = json.loads(output_fpath.read_text())
        proc_item = result['info'][-1]
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, index=1)
        return flat_resolved


class Stage1_Evaluate(ProcessNode):
    name = 'stage1_evaluate'
    executable = f'python {EXAMPLE_DPATH}/cli/stage1_evaluate.py'

    in_paths = {
        'true_fpath',
        'pred_fpath',
    }
    out_paths = {
        'out_fpath': 'stage1_evaluation.json',
    }
    algo_params = {
    }
    perf_params = {
        'workers': 0,
    }

    def load_result(self, node_dpath):
        import json
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        output_fpath = node_dpath / self.out_paths[self.primary_out_key]
        result = json.loads(output_fpath.read_text())
        proc_item = result['info'][-1]
        nest_resolved = new_process_context_parser(proc_item)
        nest_resolved['metrics'] = result['result']
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, index=1)
        return flat_resolved

    def _default_metrics2(self):
        """
        Might be renamed to default_metrics in the future.
        """
        metric_infos = [
            {
                'suffix': 'accuracy',
                'objective': 'maximize',
                'primary': True,
            },
            {
                'suffix': 'hamming_distance',
                'objective': 'minimize',
                'primary': True,
            }
        ]
        return metric_infos

    @property
    def default_vantage_points(self):
        vantage_points = [
            {
                'metric1': 'metrics.stage1_evaluate.accuracy',
                'metric2': 'metrics.stage1_evaluate.hamming_distance',
            },
        ]
        return vantage_points


def my_demo_pipeline():
    """
    Example:
        >>> from mlops_example_module.pipelines import *  # NOQA
        >>> dag = my_demo_pipeline()
        >>> dag.configure({
        ...     'stage1_predict.src_fpath': 'my-input-path',
        ... })
        >>> dag.print_graphs(shrink_labels=1, show_types=1)
        >>> queue = dag.make_queue()['queue']
        >>> queue.print_commands(with_locks=0)

    Ignore:
        from graphid import util
        proc_graph = dag.proc_graph.copy()
        util.util_graphviz.dump_nx_ondisk(proc_graph, 'proc_graph.png')
        import xdev
        xdev.startfile('proc_graph.png')
    """
    # Define the nodes as stages in the pipeline
    nodes = {}
    nodes['stage1_predict'] = Stage1_Predict()
    nodes['stage1_evaluate'] = Stage1_Evaluate()

    # Next we build the edges

    # Outputs can be connected to inputs
    nodes['stage1_predict'].outputs['dst_fpath'].connect(nodes['stage1_evaluate'].inputs['pred_fpath'])

    # Inputs can be connected to other inputs if they are reused.
    nodes['stage1_predict'].inputs['src_fpath'].connect(nodes['stage1_evaluate'].inputs['true_fpath'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag
