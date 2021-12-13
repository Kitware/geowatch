import kwarray
import ubelt as ub
import pandas as pd
import scipy
import numpy as np
import scipy.stats  # NOQA


class Result(ub.NiceRepr):
    """
    Example:
        self = Result.demo()
        print('self = {!r}'.format(self))
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
    Example:
        >>> from watch.tasks.fusion.result_analysis import *  # NOQA
        >>> self = ResultAnalysis.demo()
        >>> self.analysis()
    """

    def __init__(self, results, ignore_params=None, ignore_metrics=None):
        self.results = results
        self.ignore_params = ignore_params
        self.ignore_metrics = ignore_metrics

        self.metrics = None
        self.statistics = None

    @classmethod
    def demo(cls, num=10, rng=None):
        rng = kwarray.ensure_rng(rng)
        results = [Result.demo(rng=rng) for _ in range(num)]
        self = cls(results)
        return self

    def analysis(self):
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

        # encode if we want to maximize or minimize a metric
        metric_to_objective = {
            'ap': 'max',
            'acc': 'max',
            'f1': 'max',
            'loss': 'min',
            'brier': 'min',
        }

        if self.metrics is None:
            avail_metrics = set.intersection(*[set(r.metrics.keys()) for r in self.results])
            self.metrics = sorted(avail_metrics - set(self.ignore_metrics))

        # Analyze the impact of each parameter
        self.statistics = statistics = []
        for param_name, param_values in varied.items():
            for metric_key in self.metrics:
                stats_row = {
                    'param_name': param_name,
                    'param_values': param_values,
                    'metric': metric_key,
                }

                objective = metric_to_objective.get(metric_key, None)
                if objective is None:
                    print(f'warning assume ascending for {metric_key}')
                    objective = 'max'

                ascending = objective == 'min'

                # Find all items with this particular param value
                value_to_metric_group = {}
                value_to_metric_stats = {}
                value_to_metric = {}
                for param_value, group in table.groupby(param_name):
                    metric_group = group[['name', metric_key, param_name]]
                    metric_stats = pd.Series({
                        'mean' : metric_group[metric_key].mean(),
                        'std': metric_group[metric_key].std(),
                        'max': metric_group[metric_key].max(),
                        'min': metric_group[metric_key].min(),
                        'num': len(metric_group),
                    })
                    metric_stats['best'] = metric_stats[objective]
                    value_to_metric_stats[param_value] = metric_stats
                    value_to_metric_group[param_value] = metric_group
                    value_to_metric[param_value] = metric_group[metric_key].values

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

    def report(self):
        for metric_key in self.metrics:
            for param_name, stat_groups in ub.group_items(self.statistics, key=lambda x: x['param_name']).items():
                stats_row = ub.group_items(stat_groups, key=lambda x: x['metric'])[metric_key][0]
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
                print(ub.color_text(f'  Mean-ANOVA: p={anova_mean_p:0.4f}', 'green' if anova_rank_p < p_threshold else None))
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
