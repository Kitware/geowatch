"""
Define predict / evaluation pipelines for geowatch's MLops

To define a pipeline, the user specifies a bash executable, which parameters
correspond to inputs, outputs, algorithm settting, performance settings (perf
settings should not impact the output).

The main assumptions are:

    * All inputs and outputs are files on disk.

    * All parameters can be specified as key/value pairs (
      that includes boolean flags!)

    * File path names encode content. That is, if the content of a file
      changes, the pipeline will not detect that. We may loosen this assumption
      in the future.

The "in_paths" and "out_paths" are the most important entries to specify a
pipeline that can run. Everything else is for bookkeeping.

If the assumption of arguments as key/value pairs is broken, nodes can specify
a "command" method, where the user can define exactly what shell command to
run.
"""
from geowatch.mlops.pipeline_nodes import ProcessNode
from geowatch.mlops.pipeline_nodes import PipelineDAG
import ubelt as ub  # NOQA

PREDICT_NAME = 'pred'
EVALUATE_NAME = 'eval'


class DetectronPrediction(ProcessNode):
    """
    CommandLine:
        xdoctest -m geowatch.tasks.detectron2.pipelines DetectronPrediction

    Example:
        >>> from geowatch.tasks.detectron2.pipelines import *  # NOQA
        >>> node = DetectronPrediction()
        >>> node.configure({
        >>>     'checkpoint_fpath': 'model.pt',
        >>>     'src_fpath': 'test.kwcoco.zip',
        >>> })
        >>> command = node.command
        >>> print(node.command)
    """
    name = 'detectron_pred'
    group_dname = PREDICT_NAME

    executable = 'python -m geowatch.tasks.detectron2.predict'

    in_paths = {
        'src_fpath',
        'checkpoint_fpath',
        'model_fpath',  # should this an algo param?
    }

    out_paths = {
        'dst_fpath' : 'pred.kwcoco.zip',
    }
    primary_out_key = 'dst_fpath'

    perf_params = {
        'workers': 2,
        # 'devices': '0,',
        # 'batch_size': 1,
        # 'memmap': False,
    }

    algo_params = {
        'write_heatmap': True,
    }

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        node_type = self.name
        fpath = node_dpath / self.out_paths[self.primary_out_key]
        coco_pred_info = smart_result_parser.parse_json_header(fpath)

        coco_pred_info = smart_result_parser.parse_json_header(fpath)
        assert len(coco_pred_info) == 1
        proc_item = coco_pred_info[0]  # HACK, the name is wrong
        # proc_item = smart_result_parser.find_pred_pxl_item(coco_pred_info)
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(node_type, index=1)
        return flat_resolved


class DetectionEvaluation(ProcessNode):
    """
    CommandLine:
        xdoctest -m geowatch.tasks.detectron2.pipelines DetectionEvaluation

    Example:
        >>> from geowatch.tasks.detectron2.pipelines import *  # NOQA
        >>> node = DetectionEvaluation()
        >>> node.configure({
        >>>     'true_dataset': 'test.kwcoco.zip',
        >>>     'pred_dataset': 'poly.kwcoco.zip',
        >>> })
        >>> print(node.command)
    """
    name = 'detection_evaluation'
    executable = 'python -m kwcoco eval_detections'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_dataset',
        'pred_dataset',
    }

    out_paths = {
        'out_dpath': '.',
        'out_fpath': 'detect_metrics.json',
    }
    primary_out_key = 'out_fpath'

    algo_params = {
        'iou_thresh': 0.5,
    }

    perf_params = {
        'draw': False,
    }

    def load_result(self, node_dpath):
        from geowatch.utils import util_dotdict
        from kwcoco.metrics.confusion_measures import Measures
        import json
        import numpy as np
        fpath = node_dpath / self.out_paths[self.primary_out_key]
        data = json.loads(fpath.read_text())
        assert len(data) == 1
        item = ub.peek(data.values())
        nocls_measures = item['nocls_measures']
        nocls_measures = Measures.from_json(nocls_measures)

        maximized = nocls_measures.maximized_thresholds()
        thresh = maximized['f1']['thresh']

        keys = ['mcc', 'g1', 'f1', 'acc']
        sub_keys = ['ppv', 'tpr', 'fpr', 'tnr', 'bm', 'mk', 'f1']
        info = nocls_measures
        thresh = thresh = np.array(nocls_measures['thresholds'])
        finite_flags = np.isfinite(thresh)
        finite_thresh = thresh[finite_flags]
        for key in keys:
            if key in info:
                measure = info[key][finite_flags]
                try:
                    max_idx = np.nanargmax(measure)
                except ValueError:
                    best_thresh = np.nan
                    best_measure = np.nan
                    best_submeasures = {k: np.nan for k in sub_keys}
                    best_submeasures['thresh'] = best_thresh
                else:
                    best_thresh = float(finite_thresh[max_idx])
                    best_measure = float(measure[max_idx])
                    best_submeasures = {k: info[k][finite_flags][max_idx] for k in sub_keys}
                    best_submeasures['thresh'] = best_thresh

                best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)
                info['max_{}'.format(key)] = best_label
                info['max_{}_submeasures'.format(key)] = best_submeasures
                info['_max_{}'.format(key)] = (best_measure, best_thresh)

        at_maxf1 = {'max_f1_' + k: float(v) for k, v in info['max_f1_submeasures'].items()}

        nocls_measures['max_f1']
        metrics = ub.udict(nocls_measures) & {
            'ap', 'auc', 'nsupport', 'realpos_total', 'realneg_total',
            'trunc_auc'
        }
        nest_resolved = {}
        metrics.update(at_maxf1)
        nest_resolved['metrics'] = metrics
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, 1)
        return flat_resolved

    def _default_metrics(self):
        _display_metrics_suffixes = [
            'ap',
            'auc',
            'max_f1_f1',
            'max_f1_tpr',
            'max_f1_ppv',
        ]
        _primary_metrics_suffixes = _display_metrics_suffixes[0:2]
        return _primary_metrics_suffixes, _display_metrics_suffixes

    @property
    def default_vantage_points(self):
        vantage_points = [
            {
                'metric1': 'metrics.detection_evaluation.ap',
                'metric2': 'metrics.detection_evaluation.auc',
            },
        ]
        return vantage_points


class HeatmapEvaluation(ProcessNode):
    """
    CommandLine:
        xdoctest -m geowatch.tasks.detectron2.pipelines HeatmapEvaluation

    Example:
        >>> from geowatch.tasks.detectron2.pipelines import *  # NOQA
        >>> node = HeatmapEvaluation()
        >>> node.configure({
        >>>     'true_dataset': 'test.kwcoco.zip',
        >>>     'pred_dataset': 'pred.kwcoco.zip',
        >>> })
        >>> print(node.command)
    """
    name = 'heatmap_eval'
    executable = 'python -m geowatch.tasks.fusion.evaluate'
    # executable = 'python -m kwcoco.metrics.segmentation_metrics'

    group_dname = EVALUATE_NAME

    in_paths = {
        'true_dataset',
        'pred_dataset',
    }

    out_paths = {
        'eval_dpath': '.',
        'eval_fpath': 'pxl_eval.json',
    }
    primary_out_key = 'eval_fpath'

    algo_params = {
        'score_space': 'image',
    }

    perf_params = {
        'workers': 2,
        # These arent quite perf params.
        # They control intermediate visualization, but they don't impact
        # effective outputs so we are putting them here.
        'draw_curves': True,
        'draw_heatmaps': False,
        'viz_thresh': 'auto',
        'draw_legend': True,
        'draw_weights': False,
    }

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        fpath = node_dpath / self.out_paths[self.primary_out_key]
        info = smart_result_parser.load_pxl_eval(fpath, with_param_types=False)
        metrics = info['metrics']

        proc_item = smart_result_parser.find_pxl_eval_item(
            info['json_info']['meta']['info'])

        nest_resolved = new_process_context_parser(proc_item)
        # Hack for region ids
        nest_resolved['context']['region_ids'] = ub.Path(nest_resolved['resolved_params']['true_dataset']).name.split('.')[0]
        nest_resolved['metrics'] = metrics

        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, 1)
        return flat_resolved

    def _default_metrics(self):
        _primary_metrics_suffixes = _display_metrics_suffixes = [
            'salient_AP',
            # 'salient_APUC',
            'salient_AUC',
        ]
        return _primary_metrics_suffixes, _display_metrics_suffixes

    @property
    def default_vantage_points(self):
        vantage_points = [
            {
                'metric1': 'metrics.heatmap_eval.salient_AP',
                'metric2': 'metrics.heatmap_eval.salient_AUC',

                'scale1': 'linear',
                'scale2': 'linear',

                'objective1': 'maximize',
            },
        ]
        return vantage_points


def detectron_evaluation_pipeline():
    nodes = {}
    detectron_pred = nodes['detectron_pred'] = DetectronPrediction()
    detection_evaluation = nodes['detection_evaluation'] = DetectionEvaluation()
    heatmap_eval = nodes['heatmap_eval'] = HeatmapEvaluation()

    detectron_pred.outputs['dst_fpath'].connect(detection_evaluation.inputs['pred_dataset'])
    detectron_pred.inputs['src_fpath'].connect(detection_evaluation.inputs['true_dataset'])

    detectron_pred.outputs['dst_fpath'].connect(heatmap_eval.inputs['pred_dataset'])
    detectron_pred.inputs['src_fpath'].connect(heatmap_eval.inputs['true_dataset'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag
