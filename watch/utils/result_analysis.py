"""
Runs statistical tests on sets of configuration-metrics pairs

Example:
    >>> from watch.utils.result_analysis import *  # NOQA
    >>> # Given a list of experiments, configs, and results
    >>> # Create a ResultAnalysis object
    >>> results = ResultAnalysis([
    >>>     Result('expt0', {'param1': 2, 'param3': 'b'}, {'f1': 0.75}),
    >>>     Result('expt1', {'param1': 0, 'param3': 'c'}, {'f1': 0.92}),
    >>>     Result('expt2', {'param1': 1, 'param3': 'b'}, {'f1': 0.77}),
    >>>     Result('expt3', {'param1': 1, 'param3': 'a'}, {'f1': 0.67}),
    >>>     Result('expt4', {'param1': 0, 'param3': 'c'}, {'f1': 0.98}),
    >>>     Result('expt5', {'param1': 2, 'param3': 'a'}, {'f1': 0.86}),
    >>>     Result('expt6', {'param1': 1, 'param3': 'c'}, {'f1': 0.77}),
    >>>     Result('expt7', {'param1': 1, 'param3': 'c'}, {'f1': 0.41}),
    >>>     Result('expt8', {'param1': 1, 'param3': 'a'}, {'f1': 0.64}),
    >>>     Result('expt9', {'param1': 0, 'param3': 'b'}, {'f1': 0.95}),
    >>> ])
    >>> # Calling the analysis method prints something like the following
    >>> results.analysis()

    PARAMETER 'param1' - f1
    =======================
    f1       mean       std   max   min  num  best
    param1
    0       0.950  0.030000  0.98  0.92  3.0  0.98
    2       0.805  0.077782  0.86  0.75  2.0  0.86
    1       0.652  0.147377  0.77  0.41  5.0  0.77

    ANOVA hypothesis (roughly): the param 'param1' has no effect on the metric
        Reject this hypothesis if the p value is less than a threshold
      Rank-ANOVA: p=0.0397
      Mean-ANOVA: p=0.0277

    Pairwise T-Tests
      Is param1=0 about as good as param1=2?
        ttest_ind:  p=0.2058
      Is param1=1 about as good as param1=2?
        ttest_ind:  p=0.1508


    PARAMETER 'param3' - f1
    =======================
    f1          mean       std   max   min  num  best
    param3
    c       0.770000  0.255734  0.98  0.41  4.0  0.98
    b       0.823333  0.110151  0.95  0.75  3.0  0.95
    a       0.723333  0.119304  0.86  0.64  3.0  0.86

    ANOVA hypothesis (roughly): the param 'param3' has no effect on the metric
        Reject this hypothesis if the p value is less than a threshold
      Rank-ANOVA: p=0.5890
      Mean-ANOVA: p=0.8145

    Pairwise T-Tests
      Is param3=b about as good as param3=c?
        ttest_ind:  p=0.7266
      Is param3=a about as good as param3=b?
        ttest_ind:  p=0.3466
        ttest_rel:  p=0.3466
      Is param3=a about as good as param3=c?
        ttest_ind:  p=0.7626
"""
import kwarray
import ubelt as ub
import pandas as pd
import scipy
import numpy as np
import scipy.stats  # NOQA


class Result(ub.NiceRepr):
    """
    Storage of names, parameters, and quality metrics for a single experiment.

    Attributes:
        name (str): name of the experiment
        params (Dict[str, object]): configuration of the experiment
        metrics (Dict[str, float]): quantitative results of the experiment

    Example:
        >>> from watch.utils.result_analysis import *  # NOQA
        >>> self = Result.demo(rng=32)
        >>> print('self = {}'.format(self))
        self = <Result(name=53f57161,f1=0.33,acc=0.75,param1=1,param2=6.67,param3=a)>
    """
    def __init__(self, name, params, metrics):
        self.name = name
        self.params = params
        self.metrics = metrics

    def to_dict(self):
        row = ub.dict_union({'name': self.name}, self.metrics, self.params)
        return row

    def __nice__(self):
        row = self.to_dict()
        text = ub.repr2(row, compact=True, precision=2, sort=0)
        return text

    @classmethod
    def demo(cls, rng=None):
        import numpy as np
        import string
        rng = kwarray.ensure_rng(rng)
        demo_param_space = {
            'param1': list(range(3)),
            'param2': np.linspace(0, 10, 10),
            'param3': list(string.ascii_lowercase[0:3]),
        }
        params = {k: rng.choice(b) for k, b in demo_param_space.items()}
        metrics = {
            'f1': rng.rand(),
            'acc': rng.rand(),
        }
        name = ub.hash_data(params)[0:8]
        self = cls(name, params, metrics)
        return self


class ResultAnalysis:
    """
    Groups and runs stats on results

    Attributes:
        results (List[Result]): list of results

        ignore_metrics (Set[str]): metrics to ignore

        ignore_params (Set[str]): parameters to ignore

        metric_objectives (Dict[str, str]):
            indicate if each metrix should be maximized "max" or minimized
            "min"

        metrics (List[str]):
            only consider these metrics

    Example:
        >>> from watch.utils.result_analysis import *  # NOQA
        >>> self = ResultAnalysis.demo()
        >>> self.analysis()
    """

    def __init__(self, results, metrics=None, ignore_params=None,
                 ignore_metrics=None, metric_objectives=None):
        self.results = results
        if ignore_metrics is None:
            ignore_metrics = set()
        if ignore_params is None:
            ignore_params = set()
        self.ignore_params = ignore_params
        self.ignore_metrics = ignore_metrics

        # encode if we want to maximize or minimize a metric
        default_metric_to_objective = {
            'ap': 'max',
            'acc': 'max',
            'f1': 'max',
            #
            'loss': 'min',
            'brier': 'min',
        }
        if metric_objectives is None:
            metric_objectives = {}

        self.metric_objectives = default_metric_to_objective.copy()
        self.metric_objectives.update(metric_objectives)

        self.metrics = metrics
        self.statistics = None

    @classmethod
    def demo(cls, num=10, rng=None):
        rng = kwarray.ensure_rng(rng)
        results = [Result.demo(rng=rng) for _ in range(num)]
        self = cls(results)
        return self

    def run(self):
        self.build()
        self.report()

    def analysis(self):
        # alias for run
        return self.run()
        self.build()
        self.report()

    def build(self):
        if len(self.results) < 2:
            raise Exception('need at least 2 results')

        rows = [r.to_dict() for r in self.results]
        table = pd.DataFrame(rows)

        config_rows = [r.params for r in self.results]
        sentinel = object()
        # pd.DataFrame(config_rows).channels
        varied = ub.varied_values(config_rows, default=sentinel, min_variations=1)
        if self.ignore_params:
            for k in self.ignore_params:
                varied.pop(k, None)
        self.varied = varied

        if self.metrics is None:
            avail_metrics = set.intersection(*[set(r.metrics.keys()) for r in self.results])
            self.metrics = sorted(avail_metrics - set(self.ignore_metrics))

        # Analyze the impact of each parameter
        self.statistics = statistics = []
        for param_name in varied.keys():
            for metric_key in self.metrics:
                param_values = varied[param_name]
                stats_row = {
                    'param_name': param_name,
                    'param_values': param_values,
                    'metric': metric_key,
                }

                objective = self.metric_objectives.get(metric_key, None)
                if objective is None:
                    print(f'warning assume ascending for {metric_key=}')
                    objective = 'max'

                ascending = objective == 'min'

                # Find all items with this particular param value
                value_to_metric_group = {}
                value_to_metric_stats = {}
                value_to_metric = {}
                for param_value, group in table.groupby(param_name):
                    metric_group = group[['name', metric_key, param_name]]
                    metric_vals = metric_group[metric_key]
                    metric_vals = metric_vals.dropna()
                    metric_stats = metric_vals.describe()
                    # pd.Series({
                    #     'mean' : metric_vals.mean(),
                    #     'std': metric_vals.std(),
                    #     'max': metric_vals.max(),
                    #     'min': metric_vals.min(),
                    #     'num': len(metric_vals),
                    # })
                    metric_stats['best'] = metric_stats[objective]
                    value_to_metric_stats[param_value] = metric_stats
                    value_to_metric_group[param_value] = metric_group

                    value_to_metric[param_value] = metric_vals.values

                moments = pd.DataFrame(value_to_metric_stats).T
                moments = moments.sort_values(objective, ascending=ascending)
                moments.index.name = param_name
                moments.columns.name = metric_key

                # Determine a set of value pairs to do pairwise comparisons on
                value_pairs = ub.oset()
                value_pairs.update(map(frozenset, ub.iter_window(moments.index, 2)))
                value_pairs.update(map(frozenset, ub.iter_window(moments.sort_values('mean', ascending=ascending).index, 2)))

                # https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
                # If the researcher can make the assumptions of an identically
                # shaped and scaled distribution for all groups, except for any
                # difference in medians, then the null hypothesis is that the
                # medians of all groups are equal, and the alternative
                # hypothesis is that at least one population median of one
                # group is different from the population median of at least one
                # other group.
                try:
                    anova_krus_result = scipy.stats.kruskal(*value_to_metric.values())
                except ValueError:
                    anova_krus_result = scipy.stats.stats.KruskalResult(np.nan, np.nan)

                # https://en.wikipedia.org/wiki/One-way_analysis_of_variance
                # The One-Way ANOVA tests the null hypothesis, which states
                # that samples in all groups are drawn from populations with
                # the same mean values
                if len(value_to_metric) > 1:
                    anova_1way_result = scipy.stats.f_oneway(*value_to_metric.values())
                else:
                    anova_1way_result = scipy.stats.stats.F_onewayResult(np.nan, np.nan)

                stats_row['anova_rank_H'] = anova_krus_result.statistic
                stats_row['anova_rank_p'] = anova_krus_result.pvalue
                stats_row['anova_mean_F'] = anova_1way_result.statistic
                stats_row['anova_mean_p'] = anova_1way_result.pvalue
                stats_row['moments'] = moments

                pairwise_statistics = []
                for pair in value_pairs:
                    pair_statistics = {}
                    try:
                        param_val1, param_val2 = sorted(pair)
                    except Exception:
                        param_val1, param_val2 = (pair)
                    metric_vals1 = value_to_metric[param_val1]
                    metric_vals2 = value_to_metric[param_val2]
                    pair_statistics['value1'] = param_val1
                    pair_statistics['value2'] = param_val2
                    pair_statistics['n1'] = n1 = len(metric_vals1)
                    pair_statistics['n1'] = n2 = len(metric_vals2)
                    ttest_ind_result = scipy.stats.ttest_ind(metric_vals1, metric_vals2, equal_var=False)
                    pair_statistics['ttest_ind'] = ttest_ind_result
                    if n1 == n2:
                        ttest_rel_result = scipy.stats.ttest_rel(metric_vals1, metric_vals2)
                        pair_statistics['ttest_rel'] = ttest_rel_result
                    pairwise_statistics.append(pair_statistics)

                stats_row['pairwise'] = pairwise_statistics
                statistics.append(stats_row)

                metric_stats[metric_key] = stats_row
        self.stats_table = pd.DataFrame([
            ub.dict_diff(d, {'pairwise', 'param_values', 'moments'})
            for d in self.statistics]).sort_values('anova_rank_p')

    def report(self):
        stat_groups = ub.group_items(self.statistics, key=lambda x: x['param_name'])
        stat_groups_items = list(stat_groups.items())

        # Modify this order to change the grouping pattern
        grid = ub.named_product({
            'stat_group_item': stat_groups_items,
            'metrics': self.metrics,
        })
        for grid_item in grid:
            metric_key = grid_item['metrics']
            stat_groups_item = grid_item['stat_group_item']

            param_name, stat_group = stat_groups_item
            stats_row = ub.group_items(stat_group, key=lambda x: x['metric'])[metric_key][0]
            title = ('PARAMETER {!r} - {}'.format(param_name, metric_key))
            print('\n\n')
            print(title)
            print('=' * len(title))
            print(stats_row['moments'])
            anova_rank_p = stats_row['anova_rank_p']
            anova_mean_p = stats_row['anova_mean_p']
            # Rougly speaking
            p_threshold = 0.05
            print('')
            print(f'ANOVA hypothesis (roughly): the param {param_name!r} has no effect on the metric')
            print('    Reject this hypothesis if the p value is less than a threshold')
            print(ub.color_text(f'  Rank-ANOVA: p={anova_rank_p:0.4f}', 'green' if anova_rank_p < p_threshold else None))
            print(ub.color_text(f'  Mean-ANOVA: p={anova_mean_p:0.4f}', 'green' if anova_mean_p < p_threshold else None))
            print('')
            print('Pairwise T-Tests')
            for pairstat in stats_row['pairwise']:
                value1 = pairstat['value1']
                value2 = pairstat['value2']
                # n1 = pairstat['n1']
                print(f'  Is {param_name}={value1} about as good as {param_name}={value2}?')
                if 'ttest_ind' in pairstat:
                    ttest_ind_result = pairstat['ttest_ind']
                    print(ub.color_text(f'    ttest_ind:  p={ttest_ind_result.pvalue:0.4f}', 'green' if ttest_ind_result.pvalue < p_threshold else None))
                if 'ttest_rel' in pairstat:
                    ttest_rel_result = pairstat['ttest_ind']
                    print(ub.color_text(f'    ttest_rel:  p={ttest_rel_result.pvalue:0.4f}', 'green' if ttest_rel_result.pvalue < p_threshold else None))

        print(self.stats_table)
