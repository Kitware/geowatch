"""
This utility provides a method to define a table of hyperparameter key/values
and associated metric key/values. Given this table, and information about if
metrics are better when they are higher / lower, the ResultAnalysis class uses
several statistical methods to estimate parameter importance.

Example:
    >>> # Given a list of experiments, configs, and results
    >>> from geowatch.utils.result_analysis import ResultAnalysis
    >>> # Given a table of experiments with parameters, and metrics
    >>> table = [
    >>>     Result('expt0', {'param1': 2, 'param2': 'b'}, {'f1': 0.75, 'loss': 0.5}),
    >>>     Result('expt1', {'param1': 0, 'param2': 'c'}, {'f1': 0.92, 'loss': 0.4}),
    >>>     Result('expt2', {'param1': 1, 'param2': 'b'}, {'f1': 0.77, 'loss': 0.3}),
    >>>     Result('expt3', {'param1': 1, 'param2': 'a'}, {'f1': 0.67, 'loss': 0.2}),
    >>> ]
    >>> # Create a ResultAnalysis object and tell it what metrics should be maximized / minimized
    >>> analysis = ResultAnalysis(table, metric_objectives={'f1': 'max', 'loss': 'min'})
    >>> # An overall analysis can be obtained as follows
    >>> analysis.analysis()  # xdoctest: +IGNORE_WANT
    PARAMETER: param2 - METRIC: f1
    ==============================
    f1      count  mean       std   min    25%   50%    75%   max
    param2
    c         1.0  0.92       NaN  0.92  0.920  0.92  0.920  0.92
    b         2.0  0.76  0.014142  0.75  0.755  0.76  0.765  0.77
    a         1.0  0.67       NaN  0.67  0.670  0.67  0.670  0.67
    ...
    ANOVA: If p is low, the param 'param2' might have an effect
      Rank-ANOVA: p=0.25924026
      Mean-ANOVA: p=0.07823610
    ...
    Pairwise T-Tests
      If p is low, param2=c may outperform param2=b.
        ttest_ind:  p=nan
      If p is low, param2=b may outperform param2=a.
        ttest_ind:  p=nan
        ttest_rel:  p=nan, n_pairs=1
    ...
    PARAMETER: param1 - METRIC: loss
    ================================
    loss    count  mean       std  min    25%   50%    75%  max
    param1
    1         2.0  0.25  0.070711  0.2  0.225  0.25  0.275  0.3
    0         1.0  0.40       NaN  0.4  0.400  0.40  0.400  0.4
    2         1.0  0.50       NaN  0.5  0.500  0.50  0.500  0.5
    ...
    ANOVA: If p is low, the param 'param1' might have an effect
      Rank-ANOVA: p=0.25924026
      Mean-ANOVA: p=0.31622777
    ...
    Pairwise T-Tests
      If p is low, param1=1 may outperform 0.
        ttest_ind:  p=nan
      If p is low, param1=0 may outperform 2.
        ttest_ind:  p=nan
      param_name metric  anova_rank_H  anova_rank_p  anova_mean_F  anova_mean_p
    0     param2     f1           2.7       0.25924       81.1875      0.078236
    3     param1   loss           2.7       0.25924        4.5000      0.316228
    1     param2   loss           1.8       0.40657        0.7500      0.632456
    2     param1     f1           1.8       0.40657        2.7675      0.391181

    >>> # But specific parameters or groups of parameters can be inspected
    >>> # individually
    >>> analysis.build()
    >>> analysis.abalate(['param1'], metrics=['f1'])  # xdoctest: +IGNORE_WANT
    skillboard.ratings = {
        (0,): Rating(mu=25, sigma=8.333333333333334),
        (1,): Rating(mu=27.63523138347365, sigma=8.065506316323548),
        (2,): Rating(mu=22.36476861652635, sigma=8.065506316323548),
    }
    win_probs = {
        (0,): 0.3333333333333333,
        (1,): 0.3445959888771101,
        (2,): 0.32207067778955656,
    }
    ...
    When config(param1=1) is better than config(param1=2), the improvement in f1 is
       count  mean  std   min   25%   50%   75%   max
    0    1.0  0.02  NaN  0.02  0.02  0.02  0.02  0.02
    ...
    When config(param1=2) is better than config(param1=1), the improvement in f1 is
       count  mean  std  min  25%  50%  75%  max
    0    0.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN


Example:
    >>> # Simple example for computing a p-values between a set of baseline
    >>> # results and hypothesis you think might do better.
    >>> # Given a list of experiments, configs, and results
    >>> from geowatch.utils.result_analysis import ResultAnalysis, Result
    >>> # Given a table of experiments with parameters, and metrics
    >>> table = [
    >>>     Result('expt0', {'group': 'baseline'}, {'f1': 0.75}),
    >>>     Result('expt1', {'group': 'baseline'}, {'f1': 0.72}),
    >>>     Result('expt2', {'group': 'baseline'}, {'f1': 0.79}),
    >>>     Result('expt3', {'group': 'baseline'}, {'f1': 0.73}),
    >>>     Result('expt4', {'group': 'baseline'}, {'f1': 0.74}),
    >>>     Result('expt5', {'group': 'baseline'}, {'f1': 0.74}),
    >>>     Result('expt5', {'group': 'hypothesis'}, {'f1': 0.76}),
    >>>     Result('expt6', {'group': 'hypothesis'}, {'f1': 0.78}),
    >>>     Result('expt7', {'group': 'hypothesis'}, {'f1': 0.77}),
    >>>     Result('expt8', {'group': 'hypothesis'}, {'f1': 0.75}),
    >>> ]
    >>> # Create a ResultAnalysis object and tell it what metrics should be maximized / minimized
    >>> analysis = ResultAnalysis(table, metric_objectives={'f1': 'max'})
    >>> # An overall analysis can be obtained as follows
    >>> analysis.analysis()


This seems related to [RijnHutter2018]_. Need to look more closely to determine
its exact relation and what we can learn from it (or what we do better /
worse). Also see followup [Probst2019]_.

References:
    .. [RijnHutter2018] Hyperparameter Importance Across Datasets - https://arxiv.org/pdf/1710.04725.pdf

    .. [Probst2019] https://www.jmlr.org/papers/volume20/18-444/18-444.pdf


Look into:
    https://scikit-optimize.github.io/stable/
    https://wandb.ai/site/articles/find-the-most-important-hyperparameters-in-seconds

    https://docs.ray.io/en/latest/tune/index.html

    ray.tune

Requires:
    pip install ray
    pip install openskill


"""
import itertools as it
import math
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.stats  # NOQA
import ubelt as ub
import rich

# a list of common objectives
DEFAULT_METRIC_TO_OBJECTIVE = {
    "time": "min",
    "ap": "max",
    "acc": "max",
    "f1": "max",
    "mcc": "max",
    #
    "loss": "min",
    "brier": "min",
}


class Result(ub.NiceRepr):
    """
    Storage of names, parameters, and quality metrics for a single experiment.

    Attributes:
        name (str | None):
            Name of the experiment. Optional. This is unused in the analysis.
            (i.e. names will never be used computationally. Use them for keys)

        params (Dict[str, object]): configuration of the experiment.
            This is a dictionary mapping a parameter name to its value.

        metrics (Dict[str, float]): quantitative results of the experiment
            This is a dictionary for each quality metric computed on this
            result.

        meta (Dict | None): any other metadata about this result.
            This is unused in the analysis.

    Example:
        >>> self = Result.demo(rng=32)
        >>> print('self = {}'.format(self))
        self = <Result(name=53f57161,f1=0.33,acc=0.75,param1=1,param2=6.67,param3=a)>

    Example:
        >>> self = Result.demo(mode='alt', rng=32)
        >>> print('self = {}'.format(self))
    """

    def __init__(self, name, params, metrics, meta=None):
        self.name = name
        self.params = params
        self.metrics = metrics
        self.meta = meta

    def to_dict(self):
        row = ub.dict_union({"name": self.name}, self.metrics, self.params)
        return row

    def __nice__(self):
        row = self.to_dict()
        text = ub.urepr(row, compact=True, precision=2, sort=0)
        return text

    @classmethod
    def demo(cls, mode="null", rng=None):
        import string

        import kwarray
        import numpy as np

        rng = kwarray.ensure_rng(rng)

        if mode == "null":
            # The null hypothesis should generally be true here,
            # there is no relation between the results and parameters
            demo_param_space = {
                "param1": list(range(3)),
                "param2": np.linspace(0, 10, 10),
                "param3": list(string.ascii_lowercase[0:3]),
            }
            params = {k: rng.choice(b) for k, b in demo_param_space.items()}
            metrics = {
                "f1": rng.rand(),
                "acc": rng.rand(),
            }
        elif mode == "alt":
            # The alternative hypothesis should be true here, there is a
            # relationship between results two of the params.
            from scipy.special import expit

            params = {
                "u": rng.randint(0, 1 + 1),
                "v": rng.randint(-1, 1 + 1),
                "x": rng.randint(-2, 3 + 1),
                "y": rng.randint(-1, 2 + 1),
                "z": rng.randint(-0, 3 + 1),
            }
            noise = np.random.randn() * 1
            r = 3 * params["x"] + params["y"] ** 2 + 0.3 * params["z"] ** 3
            acc = expit(r / 20 + noise)
            metrics = {
                "acc": acc,
            }
        else:
            raise KeyError(mode)
        name = ub.hash_data(params)[0:8]
        self = cls(name, params, metrics)
        return self


class ResultTable:
    """
    An object that stores two tables of corresponding metrics and parameters.

    Helps abstract away the old Result object.

    Example:
        >>> from geowatch.utils.result_analysis import *  # NOQA
        >>> self = ResultTable.demo()
        >>> print(self.table)
    """

    def __init__(self, params, metrics):
        self.params = params
        self.metrics = metrics
        self._cache = {}

    def __len__(self):
        return len(self.params)

    @property
    def table(self):
        if 'table' not in self._cache:
            self._cache['table'] = pd.concat([self.params, self.metrics], axis=1)
        return self._cache['table']

    @property
    def result_list(self):
        if 'result_list' not in self._cache:
            new_results = [
                Result(name=f'expt_{idx:04d}', metrics=metrics, params=params)
                for idx, (metrics, params) in
                enumerate(zip(self.metrics.to_dict('records'),
                              self.params.to_dict('records')))
            ]
            self._cache['result_list'] = new_results
        return self._cache['result_list']

    @classmethod
    def demo(cls, num=10, mode="null", rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)
        results = [Result.demo(mode=mode, rng=rng) for _ in range(num)]
        self = cls.coerce(results)
        return self

    @classmethod
    def coerce(cls, data, param_cols=None, metric_cols=None):
        _cache = {}
        if isinstance(data, cls):
            return data
        elif isinstance(data, list):
            results = data
            params = pd.DataFrame([r.params for r in results])
            metrics = pd.DataFrame([r.metrics for r in results])
            _cache['result_list'] = results
        elif isinstance(data, dict):
            params = data['params']
            metrics = data['metrics']
        elif isinstance(data, pd.DataFrame):
            if param_cols is None or metric_cols is None:
                raise Exception('Both param_cols and metric_cols must be given when input is a single data frame')
            params = data[param_cols]
            metrics = data[metric_cols]
            _cache['table'] = data
        else:
            raise NotImplementedError
        self = cls(params, metrics)
        self._cache.update(_cache)
        return self

    @ub.memoize_property
    def varied(self):
        sentinel = object()
        # pd.DataFrame(config_rows).channels
        # varied = dict(varied_values(config_rows, default=sentinel, min_variations=2, dropna=True))
        varied = dict(varied_value_counts(self.params, default=sentinel, min_variations=2, dropna=True))
        # remove nans
        # varied = {
        #     k: {v for v in vs if not (isinstance(v, float) and math.isnan(v))}
        #     for k, vs in varied.items()
        # }
        varied = {
            k: {v: c for v, c in vs.items() if not (isinstance(v, float) and math.isnan(v))}
            for k, vs in varied.items()
        }
        varied = {k: vs for k, vs in varied.items() if len(vs)}
        return varied


class ResultAnalysis(ub.NiceRepr):
    """
    Groups and runs stats on results

    Runs statistical tests on sets of configuration-metrics pairs

    Attributes:
        results (List[Result] | DataFrame):
            list of results, or something coercable to one.

        ignore_metrics (Set[str]): metrics to ignore

        ignore_params (Set[str]): parameters to ignore

        metric_objectives (Dict[str, str]):
            indicate if each metrix should be maximized "max" or minimized
            "min"

        metrics (List[str]):
            only consider these metrics

        params (List[str]):
            if given, only consider these params

        abalation_orders (Set[int]):
            The number of parameters to be held constant in each statistical
            grouping. Defaults to 1, so it groups together results where 1
            variable is held constant. Including 2 will include pairwise
            settings of parameters to be held constant. Using -1 or -2 means
            all but 1 or 2 parameters will be held constant, repsectively.

        default_objective (str):
            assume max or min for unknown metrics

    Example:
        >>> self = ResultAnalysis.demo()
        >>> self.analysis()

    Example:
        >>> self = ResultAnalysis.demo(num=5000, mode='alt')
        >>> self.analysis()

    Example:
        >>> # Given a list of experiments, configs, and results
        >>> # Create a ResultAnalysis object
        >>> from geowatch.utils.result_analysis import *  # NOQA
        >>> result_table = ResultTable.coerce([
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
        >>> analysis = ResultAnalysis(result_table)
        >>> # Calling the analysis method prints something like the following
        >>> analysis.analysis()

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

    def __init__(
        self,
        results,
        metrics=None,
        params=None,
        ignore_params=None,
        ignore_metrics=None,
        metric_objectives=None,
        abalation_orders={1},
        default_objective="max",
        p_threshold=0.05,
    ):
        self.result_table = ResultTable.coerce(
            results, metric_cols=metrics, param_cols=params)

        # TODO: params_of_interest

        if ignore_metrics is None:
            ignore_metrics = set()
        if ignore_params is None:
            ignore_params = set()
        self.ignore_params = ignore_params
        self.ignore_metrics = ignore_metrics

        self.abalation_orders = abalation_orders
        self.default_objective = default_objective

        # encode if we want to maximize or minimize a metric
        if metric_objectives is None:
            metric_objectives = {}
        self.metric_objectives = DEFAULT_METRIC_TO_OBJECTIVE.copy()
        self.metric_objectives.update(metric_objectives)

        self.params = params  # todo: rename to param_cols
        self.metrics = metrics  # todo: rename to metric_cols
        self.statistics = None
        self.p_threshold = p_threshold

        self._description = {}
        self._description["built"] = False
        self._description["num_results"] = len(self.result_table)

    def __nice__(self):
        return ub.urepr(self._description, si=1, sv=1)

    @classmethod
    def demo(cls, num=10, mode="null", rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)
        results = [Result.demo(mode=mode, rng=rng) for _ in range(num)]
        if mode == "null":
            self = cls(results, metrics={"f1", "acc"})
        else:
            self = cls(results, metrics={"acc"})
        return self

    def run(self):
        self.build()
        self.report()

    def analysis(self):
        # alias for run
        return self.run()
        self.build()
        self.report()

    @property
    def table(self):
        return self.result_table.table

    def metric_table(self):
        return self.result_table.metrics

    @ub.memoize_property
    def varied(self):
        return self.result_table.varied

    def abaltion_groups(self, param_group, k=2):
        """
        Return groups where the specified parameter(s) are varied, but all
        other non-ignored parameters are held the same.

        Args:
            param_group (str | List[str]):
                One or more parameters that are allowed to vary

            k (int):
                minimum number of items a group must contain to be returned

        Returns:
            List[DataFrame]:
                a list of subsets of in the table where all but the specified
                (non-ignored) parameters are allowed to vary.

        Example:
            >>> self = ResultAnalysis.demo()
            >>> param = 'param2'
            >>> self.abaltion_groups(param)
        """
        if not ub.iterable(param_group):
            param_group = [param_group]
        table = self.table
        config_keys = [set(self.result_table.params.columns.tolist())]
        # if self.params:
        #     config_keys = list(self.params)
        if self.ignore_params:
            config_keys = [c - self.ignore_params for c in config_keys]
        isect_params = set.intersection(*config_keys)
        other_params = sorted(isect_params - set(param_group))
        groups = []
        for key, group in fix_groupby(table.groupby(other_params, dropna=False)):
            if len(group) >= k:
                groups.append(group)
        return groups

    def _objective_is_ascending(self, metric_key):
        """
        Args:
            metric_key (str): the metric in question

        Returns:
            bool:
                True if we should minimize the objective (lower is better)
                False if we should maximize the objective (higher is better)
        """
        objective = self.metric_objectives.get(metric_key, None)
        if objective is None:
            warnings.warn(f"warning assume {self.default_objective} for {metric_key=}")
            objective = self.default_objective
        ascending = objective == "min"
        return ascending

    def tune(self):
        """

        Look into:
            # Old bayes opt?
            https://github.com/Erotemic/clab/blob/master/clab/live/urban_pred.py#L459

        Example:
            >>> self = ResultAnalysis.demo(100)

        """
        from ray import tune

        # 1. Define an objective function.
        def objective(config):
            score = config["a"] ** 2 + config["b"]
            return {"score": score}

        # 2. Define a search space.
        search_space = {
            "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
            "b": tune.choice([1, 2, 3]),
        }

        # 3. Start a Tune run and print the best result.
        tuner = tune.Tuner(objective, param_space=search_space)
        results = tuner.fit()
        print(results.get_best_result(metric="score", mode="min").config)
        raise NotImplementedError

    def ablate(self, param_group, metrics=None, use_openskill='auto'):
        """
        TODO:
            rectify with test-group

        Example:
            >>> self = ResultAnalysis.demo(100)
            >>> param = 'param2'
            >>> # xdoctest: +REQUIRES(module:openskill)
            >>> self.ablate(param)

            >>> self = ResultAnalysis.demo()
            >>> param_group = ['param2', 'param3']
            >>> # xdoctest: +REQUIRES(module:openskill)
            >>> self.ablate(param_group)
        """
        if self.table is None:
            self.table = self.build_table()
        if not ub.iterable(param_group):
            param_group = [param_group]

        # For hashable generic dictionary
        from collections import namedtuple

        gd = namedtuple("config", param_group)

        # from types import SimpleNamespace
        param_unique_vals_ = (
            self.table[param_group].drop_duplicates().to_dict("records")
        )
        param_unique_vals = [gd(**d) for d in param_unique_vals_]
        # param_unique_vals = {p: self.table[p].unique().tolist() for p in param_group}
        score_improvements = ub.ddict(list)
        scored_obs = []
        if use_openskill == 'auto':
            try:
                import openskill  # NOQA
            except ImportError:
                print('warning: openskill is not installed')
                use_openskill = False
            else:
                use_openskill = True

        if use_openskill:
            skillboard = SkillTracker(param_unique_vals)
        else:
            skillboard = None

        groups = self.abalation_groups(param_group, k=2)

        if metrics is None:
            metrics = self.metrics

        if metrics is None:
            avail_metrics = set(self.result_table.metrics.columns)
            metrics_of_interest = sorted(avail_metrics - set(self.ignore_metrics))
        else:
            metrics_of_interest = metrics

        for group in groups:
            for metric_key in metrics_of_interest:
                ascending = self._objective_is_ascending(metric_key)

                group = group.sort_values(metric_key, ascending=ascending)
                subgroups = fix_groupby(group.groupby(param_group))
                if ascending:
                    best_idx = subgroups[metric_key].idxmax()
                else:
                    best_idx = subgroups[metric_key].idxmin()
                best_group = group.loc[best_idx]
                best_group = best_group.sort_values(metric_key, ascending=ascending)

                for x1, x2 in it.product(best_group.index, best_group.index):
                    if x1 != x2:
                        r1 = best_group.loc[x1]
                        r2 = best_group.loc[x2]
                        k1 = gd(**r1[param_group])
                        k2 = gd(**r2[param_group])
                        diff = r1[metric_key] - r2[metric_key]
                        score_improvements[(k1, k2, metric_key)].append(diff)

                # metric_vals = best_group[metric_key].values
                # diffs = metric_vals[None, :] - metric_vals[:, None]
                best_group.set_index(param_group)
                # best_group[param_group]
                # best_group[metric_key].diff()
                scored_ranking = best_group[param_group + [metric_key]].reset_index(
                    drop=True
                )
                scored_obs.append(scored_ranking)
                if skillboard is not None:
                    ranking = [
                        gd(**d) for d in scored_ranking[param_group].to_dict("records")
                    ]
                    skillboard.observe(ranking)

        if skillboard is not None:
            print(
                "skillboard.ratings = {}".format(
                    ub.urepr(skillboard.ratings, nl=1, align=":")
                )
            )
            win_probs = skillboard.predict_win()
            print(f"win_probs = {ub.urepr(win_probs, nl=1)}")

        for key, improves in score_improvements.items():
            k1, k2, metric_key = key
            improves = np.array(improves)
            pos_delta = improves[improves > 0]
            print(
                f"\nWhen {k1} is better than {k2}, the improvement in {metric_key} is"
            )
            print(pd.DataFrame([pd.Series(pos_delta).describe().T]))
        return scored_obs

    abalation_groups = abaltion_groups
    abalate = ablate

    def test_group(self, param_group, metric_key):
        """
        Get stats for a particular metric / constant group

        Args:
            param_group (List[str]): group of parameters to hold constant.
            metric_key (str): The metric to test.

        Returns:
            dict
            # TODO : document these stats clearly and accurately

        Example:
            >>> self = ResultAnalysis.demo(num=100)
            >>> print(self.table)
            >>> param_group = ['param2', 'param1']
            >>> metric_key = 'f1'
            >>> stats_row = self.test_group(param_group, metric_key)
            >>> print('stats_row = {}'.format(ub.urepr(stats_row, nl=2, sort=0, precision=2)))
        """
        param_group_name = ",".join(param_group)
        stats_row = {
            "param_name": param_group_name,
            "metric": metric_key,
        }
        # param_values = varied[param_name]
        # stats_row['param_values'] = param_values
        ascending = self._objective_is_ascending(metric_key)

        # Find all items with this particular param value
        value_to_metric_group = {}
        value_to_metric_stats = {}
        value_to_metric = {}

        varied_cols = sorted(self.varied.keys())

        # Not sure if this is the right name, these are the other param keys
        # that we are not directly investigating, but might have an impact.
        # We use these to select comparable rows for pairwise t-tests
        nuisance_cols = sorted(set(self.varied.keys()) - set(param_group))

        grouper = fix_groupby(self.table.groupby(param_group))
        for param_value, group in grouper:
            metric_group = group[[metric_key] + varied_cols]
            metric_vals = metric_group[metric_key]
            metric_vals = metric_vals.dropna()
            if len(metric_vals) > 0:
                metric_stats = metric_vals.describe()
                value_to_metric_stats[param_value] = metric_stats
                value_to_metric_group[param_value] = metric_group
                value_to_metric[param_value] = metric_vals.values

        moments = pd.DataFrame(value_to_metric_stats).T
        if "mean" not in moments.columns:
            raise ValueError(f'No values for {metric_key}')

        moments = moments.sort_values("mean", ascending=ascending)
        moments.index.name = param_group_name
        moments.columns.name = metric_key
        ranking = moments["mean"].index.to_list()
        param_to_rank = ub.invert_dict(dict(enumerate(ranking)))

        # Determine a set of value pairs to do pairwise comparisons on
        value_pairs = ub.oset()
        # value_pairs.update(
        #     map(frozenset, ub.iter_window(moments.index, 2)))
        value_pairs.update(
            map(
                frozenset,
                ub.iter_window(
                    moments.sort_values("mean", ascending=ascending).index, 2
                ),
            )
        )

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

        stats_row["anova_rank_H"] = anova_krus_result.statistic
        stats_row["anova_rank_p"] = anova_krus_result.pvalue
        stats_row["anova_mean_F"] = anova_1way_result.statistic
        stats_row["anova_mean_p"] = anova_1way_result.pvalue
        stats_row["moments"] = moments

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero', category=RuntimeWarning)
            warnings.filterwarnings('ignore', 'invalid value', category=RuntimeWarning)

            pair_stats_list = []
            for pair in value_pairs:
                pair_stats = {}
                param_val1, param_val2 = pair

                metric_vals1 = value_to_metric[param_val1]
                metric_vals2 = value_to_metric[param_val2]

                rank1 = param_to_rank[param_val1]
                rank2 = param_to_rank[param_val2]
                pair_stats["winner"] = param_val1 if rank1 < rank2 else param_val2
                pair_stats["value1"] = param_val1
                pair_stats["value2"] = param_val2
                pair_stats["n1"] = len(metric_vals1)
                pair_stats["n2"] = len(metric_vals2)

                TEST_ONLY_FOR_DIFFERENCE = True
                if TEST_ONLY_FOR_DIFFERENCE:
                    if ascending:
                        # We want to minimize the metric
                        alternative = "less" if rank1 < rank2 else "greater"
                    else:
                        # We want to maximize the metric
                        alternative = "greater" if rank1 < rank2 else "less"
                else:
                    alternative = "two-sided"

                ind_kw = dict(
                    equal_var=False,
                    alternative=alternative,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Degrees of freedom', category=RuntimeWarning)
                    warnings.filterwarnings('ignore', 'invalid value', category=RuntimeWarning)
                    ttest_ind_result = scipy.stats.ttest_ind(
                        metric_vals1, metric_vals2, **ind_kw
                    )

                if 0:
                    from benchmarker.benchmarker import stats_dict

                    stats1 = stats_dict(metric_vals1)
                    stats2 = stats_dict(metric_vals2)
                    scipy.stats.ttest_ind_from_stats(
                        stats1["mean"],
                        stats1["std"],
                        stats1["nobs"],
                        stats2["mean"],
                        stats2["std"],
                        stats2["nobs"],
                        **ind_kw,
                    )
                    # metric_vals1, metric_vals2, equal_var=False)

                scipy.stats.ttest_ind_from_stats

                pair_stats["ttest_ind"] = ttest_ind_result

                # Do relative checks, need to find comparable subgroups
                metric_group1 = value_to_metric_group[param_val1]
                metric_group2 = value_to_metric_group[param_val2]
                if nuisance_cols:
                    nuisance_vals1 = metric_group1[nuisance_cols]
                    nuisance_vals2 = metric_group2[nuisance_cols]
                    nk_to_group1 = dict(list(fix_groupby(nuisance_vals1.groupby(nuisance_cols))))
                    nk_to_group2 = dict(list(fix_groupby(nuisance_vals2.groupby(nuisance_cols))))
                else:
                    nk_to_group1 = {None: metric_group1}
                    nk_to_group2 = {None: metric_group2}
                common = set(nk_to_group1.keys()) & set(nk_to_group2.keys())
                comparable_indexes1 = []
                comparable_indexes2 = []
                if common:
                    for nk in common:
                        group1 = nk_to_group1[nk]
                        group2 = nk_to_group2[nk]
                        # TODO: Not sure if taking the product of everything within
                        # the comparable group is correct or not. I think it is ok.
                        # TODO: randomly take these if there are too many
                        for i, j in it.product(group1.index, group2.index):
                            comparable_indexes1.append(i)
                            comparable_indexes2.append(j)

                    comparable_groups1 = metric_group1.loc[comparable_indexes1, metric_key]
                    comparable_groups2 = metric_group2.loc[comparable_indexes2, metric_key]

                    # Does this need to have the values aligned?
                    # I think that is the case giving my understanding of paired
                    # t-tests, but the docs need a PR to make that more clear.
                    ttest_rel_result = scipy.stats.ttest_rel(
                        comparable_groups1, comparable_groups2
                    )
                    pair_stats["n_common"] = len(common)
                    pair_stats["ttest_rel"] = ttest_rel_result
                pair_stats_list.append(pair_stats)

        stats_row["pairwise"] = pair_stats_list
        return stats_row

    def build(self):
        import itertools as it

        if len(self.result_table) < 2:
            raise Exception("need at least 2 results")

        varied = self.varied.copy()
        if self.ignore_params:
            for k in self.ignore_params:
                varied.pop(k, None)
        if self.params:
            varied = ub.dict_isect(varied, self.params)

        # Experimental:
        # Find Auto-abalation groups
        # TODO: when the group size is -1, instead of showing all of the group
        # settings, for each group setting do the k=1 analysis within that group
        varied_param_names = list(varied.keys())
        num_varied_params = len(varied)
        held_constant_orders = {
            num_varied_params + i if i < 0 else i for i in self.abalation_orders
        }
        held_constant_orders = [i for i in held_constant_orders if i > 0]
        held_constant_groups = []
        for k in held_constant_orders:
            held_constant_groups.extend(
                list(map(list, it.combinations(varied_param_names, k)))
            )

        if self.metrics is None:
            avail_metrics = set(self.result_table.metrics.columns)
            metrics_of_interest = sorted(avail_metrics - set(self.ignore_metrics))
        else:
            metrics_of_interest = self.metrics
        self.metrics_of_interest = metrics_of_interest
        self._description["metrics_of_interest"] = metrics_of_interest
        self._description["num_groups"] = len(held_constant_groups)

        # Analyze the impact of each parameter
        self.statistics = statistics = []
        for param_group in held_constant_groups:
            for metric_key in metrics_of_interest:
                try:
                    stats_row = self.test_group(param_group, metric_key)
                except ValueError as ex:
                    warnings.warn(repr(ex))
                    # print(f'param_group={param_group}')
                    # print(f'metric_key={metric_key}')
                    # raise
                else:
                    statistics.append(stats_row)

        self.stats_table = pd.DataFrame(
            [
                ub.dict_diff(d, {"pairwise", "param_values", "moments"})
                for d in self.statistics
            ]
        )

        if len(self.stats_table):
            self.stats_table = self.stats_table.sort_values("anova_rank_p")

        self._description["built"] = True

    def report(self):
        stat_groups = ub.group_items(self.statistics, key=lambda x: x["param_name"])
        stat_groups_items = list(stat_groups.items())

        # Modify this order to change the grouping pattern
        grid = ub.named_product(
            {
                "stat_group_item": stat_groups_items,
                "metrics": self.metrics_of_interest,
            }
        )
        for grid_item in grid:
            ...
            self._report_one(grid_item)

        # print(self.stats_table)
        rich.print(self.stats_table.to_string())

    def _report_one(self, grid_item):
        p_threshold = self.p_threshold
        metric_key = grid_item["metrics"]
        stat_groups_item = grid_item["stat_group_item"]

        param_name, stat_group = stat_groups_item
        param_name_show = ub.color_text(param_name, color='yellow')
        metric_key_show =  ub.color_text(metric_key, color='blue')
        stats_row = ub.group_items(stat_group, key=lambda x: x["metric"])[metric_key][0]
        title = f"PARAMETER: {param_name_show} - METRIC: {metric_key_show}"
        print("\n\n")
        print(title)
        print("=" * len(title))
        print(stats_row["moments"])
        anova_rank_p = stats_row["anova_rank_p"]
        anova_mean_p = stats_row["anova_mean_p"]
        # Rougly speaking
        print("")
        print(f"ANOVA: If p is low, the param {param_name_show} might have an effect")
        print(
            ub.color_text(
                f"  Rank-ANOVA: p={anova_rank_p:0.8f}",
                "green" if anova_rank_p < p_threshold else None,
            )
        )
        print(
            ub.color_text(
                f"  Mean-ANOVA: p={anova_mean_p:0.8f}",
                "green" if anova_mean_p < p_threshold else None,
            )
        )
        print("")
        print("Pairwise T-Tests")
        for pairstat in stats_row["pairwise"]:
            # Is this backwards?
            value1 = pairstat["value1"]
            value2 = pairstat["value2"]
            winner = pairstat["winner"]
            if value2 == winner:
                value1, value2 = value2, value1
            print(
                f"  If p is low, {value1} may outperform {value2} for {param_name_show}."
            )
            if "ttest_ind" in pairstat:
                ttest_ind_result = pairstat["ttest_ind"]
                print(
                    ub.color_text(
                        f"    ttest_ind:  p={ttest_ind_result.pvalue:0.8f}",
                        "green" if ttest_ind_result.pvalue < p_threshold else None,
                    )
                )
            if "ttest_rel" in pairstat:
                n_common = pairstat["n_common"]
                ttest_rel_result = pairstat["ttest_ind"]
                print(
                    ub.color_text(
                        f"    ttest_rel:  p={ttest_rel_result.pvalue:0.8f}, n_pairs={n_common}",
                        "green" if ttest_rel_result.pvalue < p_threshold else None,
                    )
                )

    def conclusions(self):
        conclusions = []
        for stat in self.statistics:
            param_name = stat["param_name"]
            metric = stat["metric"]
            for pairstat in stat["pairwise"]:
                value1 = pairstat["value1"]
                value2 = pairstat["value2"]
                winner = pairstat["winner"]
                if value2 == winner:
                    value1, value2 = value2, value1
                pvalue = stat = pairstat["ttest_ind"].pvalue
                if round(pvalue, 8) == 0:
                    txt = f"p={pvalue:0.2g}, If p is low, {value1} may outperform {value2} for {param_name} on {metric}."
                else:
                    txt = f"p={pvalue:0.8f}, If p is low, {value1} may outperform {value2} for {param_name} on {metric}."
                conclusions.append(txt)
        return conclusions

    def plot(self, xlabel, metric_key, group_labels, data=None, **kwargs):
        """
        Args:
            group_labels (dict):
                Tells seaborn what attributes to use to distinsuish curves like
                hue, size, marker. Also can contain "col" for use with
                FacetGrid, and "fig" to separate different configurations into
                different figures.

        Returns:
            List[Dict]:
                A list for each figure containing info abou that figure for any
                postprocessing.

        Example:
            >>> self = ResultAnalysis.demo(num=1000, mode='alt')
            >>> self.analysis()
            >>> print('self = {}'.format(self))
            >>> print('self.varied = {}'.format(ub.urepr(self.varied, nl=1)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.autosns()
            >>> xlabel = 'x'
            >>> metric_key = 'acc'
            >>> group_labels = {
            >>>     'fig': ['u'],
            >>>     'col': ['y', 'v'],
            >>>     'hue': ['z'],
            >>>     'size': [],
            >>> }
            >>> kwargs = {'xscale': 'log', 'yscale': 'log'}
            >>> self.plot(xlabel, metric_key, group_labels, **kwargs)
        """
        print("Init seaborn and pyplot")
        import seaborn as sns

        sns.set()
        from matplotlib import pyplot as plt  # NOQA

        print("Starting plot")

        if data is None:
            data = self.table
        data = data.sort_values(metric_key)

        print("Compute group labels")
        for gname, labels in group_labels.items():
            if len(labels):
                new_col = []
                for row in data[labels].to_dict("records"):
                    item = ub.urepr(row, compact=1, si=1)
                    new_col.append(item)
                gkey = gname + "_key"
                data[gkey] = new_col

        plot_kws = {
            "x": xlabel,
            "y": metric_key,
        }
        for gname, labels in group_labels.items():
            if labels:
                plot_kws[gname] = gname + "_key"

        # Your variables may change
        # ax = plt.figure().gca()
        fig_params = plot_kws.pop("fig", [])

        facet_kws = {
            "sharex": True,
            "sharey": True,
        }
        # facet_kws['col'] = plot_kws.pop("col", None)
        # facet_kws['row'] = plot_kws.pop("row", None)
        # if not facet_kws['row']:
        #     facet_kws['col_wrap'] = 5
        plot_kws["row"] = plot_kws.get("row", None)
        # if not plot_kws['row']:
        #     plot_kws['col_wrap'] = 5

        if not fig_params:
            groups = [("", data)]
        else:
            groups = fix_groupby(data.groupby(fig_params))

        if "marker" not in plot_kws:
            plot_kws["marker"] = "o"

        # We will want to overwrite this with our own std estimate
        plot_kws["ci"] = "sd"
        # err_style='band',
        # err_kws=None,

        # Use a consistent pallete across plots
        unique_hues = data["hue_key"].unique()
        palette = ub.dzip(unique_hues, sns.color_palette(n_colors=len(unique_hues)))
        plot_kws["palette"] = palette

        # kwplot.close_figures()

        plots = []
        base_fnum = 1
        print("Start plots")
        # hack
        hack_groups = [(k, v) for k, v in groups if k != "input=Complex object"]

        for fnum, (fig_key, group) in enumerate(hack_groups, start=base_fnum):
            # TODO: seaborn doesn't give us any option to reuse an existing
            # figure or even specify what it's handle should be. A patch should
            # be submitted to add that feature, but in the meantime work around
            # it and use the figures they give us.

            # fig = plt.figure(fnum)
            # fig.clf()

            facet = sns.relplot(
                data=group,
                kind="line",
                # kind="scatter",
                facet_kws=facet_kws,
                **plot_kws,
            )
            # See ~/code/ultrajson/json_benchmarks/benchmarker/util_stats.py
            # from json_benchmarks.benchmarker.util_stats import aggregate_stats

            facet_data_groups = dict(list(fix_groupby(facet.data.groupby(facet._col_var))))
            # facet_data_group_iter = iter(facet_data_groups.keys())

            for ax in facet.axes.ravel():
                col_key = ax.get_title().split("=", 1)[-1].strip()
                # col_key = next(facet_data_group_iter)
                col_data = facet_data_groups[col_key]
                col_data["mean_time"]
                col_data["std_time"]
                xlabel = plot_kws["x"]
                ylabel = plot_kws["y"]
                subgroups = fix_groupby(col_data.groupby(plot_kws["hue"]))
                for subgroup_key, subgroup in subgroups:
                    # combine stds in multiple groups on the x and manually draw errors
                    suffix = "_" + ylabel.partition("_")[2]
                    if "mean_" in ylabel:
                        std_label = ylabel.replace("mean_", "std_")
                        combo_group = aggregate_stats(
                            subgroup, suffix=suffix, group_keys=[plot_kws["x"]]
                        )
                        _xdata = combo_group[xlabel].values
                        _ydata_mean = combo_group[ylabel].values
                        _ydata_std = combo_group[std_label].values
                        std_label = ylabel.replace("mean_", "std_")

                        # Plot bars 3 standard deviations from the mean to
                        # get a 99.7% interval
                        num_std = 3
                        y_data_min = _ydata_mean - num_std * _ydata_std
                        y_data_max = _ydata_mean + num_std * _ydata_std
                        spread_alpha = 0.3
                        color = palette[subgroup_key]
                        ax.fill_between(
                            _xdata,
                            y_data_min,
                            y_data_max,
                            alpha=spread_alpha,
                            color=color,
                            zorder=1,
                        )
                    # zorder=0)

            xscale = kwargs.get("xscale", None)
            yscale = kwargs.get("yscale", None)
            for ax in facet.axes.ravel():
                if xscale is not None:
                    try:
                        ax.set_xscale(xscale)
                    except ValueError:
                        pass
                if yscale is not None:
                    try:
                        ax.set_yscale(yscale)
                    except ValueError:
                        pass

            fig = facet.figure
            fig.suptitle(fig_key)
            fig.tight_layout()
            # facet = sns.FacetGrid(group, **facet_kws)
            # facet.map_dataframe(sns.lineplot, x=xlabel, y=metric_key, **plot_kws)
            # facet.add_legend()

            plot = {
                "fig": fig,
                "facet": facet,
            }
            plots.append(plot)

            # if fnum >= 1:
            #     break

        # print("Adjust plots")
        # for plot in plots:
        #     xscale = kwargs.get("xscale", None)
        #     yscale = kwargs.get("yscale", None)
        #     facet = plot["facet"]

        #     facet_data_groups = dict(list(facet.data.groupby(facet._col_var)))
        #     facet_data_group_iter = iter(facet_data_groups.keys())

        #     for ax in facet.axes.ravel():

        #         if xscale is not None:
        #             try:
        #                 ax.set_xscale(xscale)
        #             except ValueError:
        #                 pass
        #         if yscale is not None:
        #             try:
        #                 ax.set_yscale(yscale)
        #             except ValueError:
        #                 pass
        print("Finish")
        return plots


class SkillTracker:
    """
    Wrapper around openskill

    Args:
        player_ids (List[T]):
            a list of ids (usually ints) used to represent each player

    Example:
        >>> # xdoctest: +REQUIRES(module:openskill)
        >>> self = SkillTracker([1, 2, 3, 4, 5])
        >>> self.observe([2, 3])  # Player 2 beat player 3.
        >>> self.observe([1, 2, 5, 3])  # Player 3 didnt play this round.
        >>> self.observe([2, 3, 4, 5, 1])  # Everyone played, player 2 won.
        >>> win_probs = self.predict_win()
        >>> print('win_probs = {}'.format(ub.urepr(win_probs, nl=1, precision=2)))
        win_probs = {
            1: 0.20,
            2: 0.21,
            3: 0.19,
            4: 0.20,
            5: 0.20,
        }

    Requirements:
        openskill
    """

    def __init__(self, player_ids):
        import openskill

        self.player_ids = player_ids
        self.ratings = {m: openskill.Rating() for m in player_ids}
        # self.observations = []

    def predict_win(self):
        """
        Estimate the probability that a particular player will win given the
        current ratings.

        Returns:
            Dict[T, float]: mapping from player ids to win probabilites
        """
        from openskill import predict_win

        teams = [[p] for p in list(self.ratings.keys())]
        ratings = [[r] for r in self.ratings.values()]
        probs = predict_win(ratings)
        win_probs = {team[0]: prob for team, prob in zip(teams, probs)}
        return win_probs

    def observe(self, ranking):
        """
        After simulating a round, pass the ranked order of who won
        (winner is first, looser is last) to this function. And it
        updates the rankings.

        Args:
            ranking (List[T]):
                ranking of all the players that played in this round
                winners are at the front (0-th place) of the list.
        """
        import openskill

        # self.observations.append(ranking)
        ratings = self.ratings
        team_standings = [[r] for r in ub.take(ratings, ranking)]
        # new_values = openskill.rate(team_standings)  # Not inplace
        # new_ratings = [openskill.Rating(*new[0]) for new in new_values]
        new_team_ratings = openskill.rate(team_standings)
        new_ratings = [new[0] for new in new_team_ratings]
        ratings.update(ub.dzip(ranking, new_ratings))


class UnhashablePlaceholder(str):
    ...


def varied_values(longform, min_variations=0, max_variations=None,
                  default=ub.NoParam, dropna=False, on_error='raise'):
    """
    Given a list of dictionaries, find the values that differ between them.

    Args:
        longform (List[Dict[KT, VT]] | DataFrame):
            This is longform data, as described in [SeabornLongform]_. It is a
            list of dictionaries.

            Each item in the list - or row - is a dictionary and can be thought
            of as an observation. The keys in each dictionary are the columns.
            The values of the dictionary must be hashable. Lists will be
            converted into tuples.

        min_variations (int, default=0):
            "columns" with fewer than ``min_variations`` unique values are
            removed from the result.

        max_variations (int | None):
            If specified only return items with fewer than this number of
            variations.

        default (VT | NoParamType):
            if specified, unspecified columns are given this value.
            Defaults to NoParam.

        on_error (str):
            Error policy when trying to add a non-hashable type.
            Default to "raise". Can be "raise", "ignore", or "placeholder",
            which will impute a hashable error message.

    Returns:
        Dict[KT, List[VT]] :
            a mapping from each "column" to the set of unique values it took
            over each "row". If a column is not specified for each row, it is
            assumed to take a `default` value, if it is specified.

    Raises:
        KeyError: If ``default`` is unspecified and all the rows
            do not contain the same columns.

    References:
        .. [SeabornLongform] https://seaborn.pydata.org/tutorial/data_structure.html#long-form-data
    """
    # Enumerate all defined columns
    import numbers

    if isinstance(longform, pd.DataFrame):
        longform = longform.to_dict('records')

    columns = set()
    for row in longform:
        if default is ub.NoParam and len(row) != len(columns) and len(columns):
            missing = set(columns).symmetric_difference(set(row))
            raise KeyError((
                'No default specified and not every '
                'row contains columns {}').format(missing))
        columns.update(row.keys())

    cannonical_nan = float('nan')

    # Build up the set of unique values for each column
    varied = ub.ddict(set)
    for row in longform:
        for key in columns:
            value = row.get(key, default)
            if isinstance(value, list):
                value = tuple(value)
            if isinstance(value, numbers.Number) and math.isnan(value):
                if dropna:
                    continue
                else:
                    # Always use a single nan value such that the id check
                    # passes. Otherwise we could end up with a dictionary that
                    # contains multiple nan keys.
                    # References:
                    # .. [SO6441857] https://stackoverflow.com/questions/6441857/nans-as-key-in-dictionaries
                    value = cannonical_nan
            try:
                varied[key].add(value)
            except TypeError as ex:
                if on_error == 'raise':
                    error_note = f'key={key}, {value}={value}'
                    if hasattr(ex, 'add_note'):
                        # Requires python.311 PEP 678
                        ex.add_note(error_note)
                        raise
                    else:
                        raise type(ex)(str(ex) + chr(10) + error_note)
                elif on_error == 'placeholder':
                    varied[key].add(UnhashablePlaceholder(value))
                elif on_error == 'ignore':
                    ...
                else:
                    raise KeyError(on_error)

    # Remove any column that does not have enough variation
    if min_variations > 0:
        for key, values in list(varied.items()):
            if len(values) < min_variations:
                varied.pop(key)

    if max_variations is not None:
        for key, values in list(varied.items()):
            if len(values) > max_variations:
                varied.pop(key)

    return varied


def varied_value_counts(longform, min_variations=0, max_variations=None,
                        default=ub.NoParam, dropna=False, on_error='raise'):
    """
    Given a list of dictionaries, find the values that differ between them.

    Args:
        longform (List[Dict[KT, VT]] | DataFrame):
            This is longform data, as described in [SeabornLongform]_. It is a
            list of dictionaries.

            Each item in the list - or row - is a dictionary and can be thought
            of as an observation. The keys in each dictionary are the columns.
            The values of the dictionary must be hashable. Lists will be
            converted into tuples.

        min_variations (int):
            "columns" with fewer than ``min_variations`` unique values are
            removed from the result. Defaults to 0.

        max_variations (int | None):
            If specified only return items with fewer than this number of
            variations.

        default (VT | NoParamType):
            if specified, unspecified columns are given this value.
            Defaults to NoParam.

        on_error (str):
            Error policy when trying to add a non-hashable type.
            Default to "raise". Can be "raise", "ignore", or "placeholder",
            which will impute a hashable error message.

    Returns:
        Dict[KT, Dict[VT, int]] :
            a mapping from each "column" to the set of unique values it took
            over each "row" and how many times it took that value. If a column
            is not specified for each row, it is assumed to take a `default`
            value, if it is specified.

    Raises:
        KeyError: If ``default`` is unspecified and all the rows
            do not contain the same columns.

    References:
        .. [SeabornLongform] https://seaborn.pydata.org/tutorial/data_structure.html#long-form-data

    Example:
        longform = [
            {'a': 'on',  'b': 'red'},
            {'a': 'on',  'b': 'green'},
            {'a': 'off', 'b': 'blue'},
            {'a': 'off', 'b': 'black'},
        ]
    """
    # Enumerate all defined columns
    import numbers

    if isinstance(longform, pd.DataFrame):
        longform = longform.to_dict('records')

    columns = set()
    for row in longform:
        if default is ub.NoParam and len(row) != len(columns) and len(columns):
            missing = set(columns).symmetric_difference(set(row))
            raise KeyError((
                'No default specified and not every '
                'row contains columns {}').format(missing))
        columns.update(row.keys())

    cannonical_nan = float('nan')

    # Build up the set of unique values for each column
    from collections import Counter
    varied_counts = ub.ddict(Counter)
    for row in longform:
        for key in columns:
            value = row.get(key, default)
            if isinstance(value, list):
                value = tuple(value)

            if isinstance(value, numbers.Number) and math.isnan(value):
                if dropna:
                    continue
                else:
                    # Always use a single nan value such that the id check
                    # passes. Otherwise we could end up with a dictionary that
                    # contains multiple nan keys.
                    # References:
                    # .. [SO6441857] https://stackoverflow.com/questions/6441857/nans-as-key-in-dictionaries
                    value = cannonical_nan
            try:
                varied_counts[key][value] += 1
            except TypeError as ex:
                if on_error == 'raise':
                    error_note = f'key={key}, {value}={value}'
                    if hasattr(ex, 'add_note'):
                        # Requires python.311 PEP 678
                        ex.add_note(error_note)
                        raise
                    else:
                        raise type(ex)(str(ex) + chr(10) + error_note)
                elif on_error == 'placeholder':
                    varied_counts[key][UnhashablePlaceholder(value)] += 1
                elif on_error == 'ignore':
                    ...
                else:
                    raise KeyError(on_error)

    # Remove any column that does not have enough variation
    if min_variations > 0:
        for key, values in list(varied_counts.items()):
            if len(values) < min_variations:
                varied_counts.pop(key)

    if max_variations is not None:
        for key, values in list(varied_counts.items()):
            if len(values) > max_variations:
                varied_counts.pop(key)

    return varied_counts


if 1:
    # Note: Code is duplicated in util_pandas
    # Fix pandas groupby so it uses the new behavior with a list of len 1
    import wrapt

    class GroupbyFutureWrapper(wrapt.ObjectProxy):
        """
        Wraps a groupby object to get the new behavior sooner.
        """

        def __iter__(self):
            keys = self.keys
            if isinstance(keys, list) and len(keys) == 1:
                # Handle this special case to avoid a warning
                for key, group in self.grouper.get_iterator(self._selected_obj, axis=self.axis):
                    yield (key,), group
            else:
                # Otherwise use the parent impl
                yield from self.__wrapped__.__iter__()

    def fix_groupby(groups):
        keys = groups.keys
        if isinstance(keys, list) and len(keys) == 1:
            return GroupbyFutureWrapper(groups)
        else:
            return groups

# import xdev
# xdev.make_warnings_print_tracebacks()


def aggregate_stats(data, suffix="", group_keys=None):
    """
    Given columns interpreted as containing stats, aggregate those stats
    within each group. For each row, any non-group, non-stat column
    with consistent values across that columns in the group is kept as-is,
    otherwise the new column for that row is set to None.

    Args:
        data (DataFrame):
            a data frame with columns: 'mean', 'std', 'min', 'max', and 'nobs'
            (possibly with a suffix)

        suffix (str):
            if the nobs, std, mean, min, and max have a suffix, specify it

        group_keys (List[str]):
            pass

    Returns:
        DataFrame:
            New dataframe where grouped rows have been aggregated into a single
            row.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame([
        >>>     #
        >>>     {'mean': 8, 'std': 1, 'min': 0, 'max': 1, 'nobs': 2, 'p1': 'a', 'p2': 1},
        >>>     {'mean': 6, 'std': 2, 'min': 0, 'max': 1, 'nobs': 3, 'p1': 'a', 'p2': 1},
        >>>     {'mean': 7, 'std': 3, 'min': 0, 'max': 2, 'nobs': 5, 'p1': 'a', 'p2': 2},
        >>>     {'mean': 5, 'std': 4, 'min': 0, 'max': 3, 'nobs': 7, 'p1': 'a', 'p2': 1},
        >>>     #
        >>>     {'mean': 3, 'std': 1, 'min': 0, 'max': 20, 'nobs': 6, 'p1': 'b', 'p2': 1},
        >>>     {'mean': 0, 'std': 2, 'min': 0, 'max': 20, 'nobs': 26, 'p1': 'b', 'p2': 2},
        >>>     {'mean': 9, 'std': 3, 'min': 0, 'max': 20, 'nobs': 496, 'p1': 'b', 'p2': 1},
        >>>     #
        >>>     {'mean': 5, 'std': 0, 'min': 0, 'max': 1, 'nobs': 2, 'p1': 'c', 'p2': 2},
        >>>     {'mean': 5, 'std': 0, 'min': 0, 'max': 1, 'nobs': 7, 'p1': 'c', 'p2': 2},
        >>>     #
        >>>     {'mean': 5, 'std': 2, 'min': 0, 'max': 2, 'nobs': 7, 'p1': 'd', 'p2': 2},
        >>>     #
        >>>     {'mean': 5, 'std': 2, 'min': 0, 'max': 2, 'nobs': 7, 'p1': 'e', 'p2': 1},
        >>> ])
        >>> print(data)
        >>> new_data = aggregate_stats(data)
        >>> print(new_data)
        >>> new_data1 = aggregate_stats(data, group_keys=['p1'])
        >>> print(new_data1)
        >>> new_data2 = aggregate_stats(data, group_keys=['p2'])
        >>> print(new_data2)
    """
    import pandas as pd

    # Stats groupings
    raw_stats_cols = ["nobs", "std", "mean", "max", "min"]
    stats_cols = [c + suffix for c in raw_stats_cols]
    mapper = dict(zip(stats_cols, raw_stats_cols))
    unmapper = dict(zip(raw_stats_cols, stats_cols))
    non_stats_cols = list(ub.oset(data.columns) - stats_cols)
    if group_keys is None:
        group_keys = non_stats_cols
    non_group_keys = list(ub.oset(non_stats_cols) - group_keys)

    new_rows = []
    for group_vals, group in list(data.groupby(group_keys)):
        # hack, is this a pandas bug in 1.4.1? Is it fixed? (Not in 1.4.2)
        if isinstance(group_keys, list) and len(group_keys) == 1:
            # For some reason, when we specify group keys as a list of one
            # element, we get a squeezed value out
            group_vals = (group_vals,)
        stat_data = group[stats_cols].rename(mapper, axis=1)
        new_stats = combine_stats_arrs(stat_data)
        new_time_stats = ub.map_keys(unmapper, new_stats)
        new_row = ub.dzip(group_keys, group_vals)
        if non_group_keys:
            for k in non_group_keys:
                unique_vals = group[k].unique()
                if len(unique_vals) == 1:
                    new_row[k] = unique_vals[0]
                else:
                    new_row[k] = None
        new_row.update(new_time_stats)
        new_rows.append(new_row)
    new_data = pd.DataFrame(new_rows)
    return new_data


def stats_dict(data, suffix=""):
    stats = {
        "nobs" + suffix: len(data),
        "mean" + suffix: data.mean(),
        "std" + suffix: data.std(),
        "min" + suffix: data.min(),
        "max" + suffix: data.max(),
    }
    return stats


def combine_stats(s1, s2):
    """
    Helper for combining mean and standard deviation of multiple measurements

    Args:
        s1 (dict): stats dict containing mean, std, and n
        s2 (dict): stats dict containing mean, std, and n

    Example:
        >>> from geowatch.utils.result_analysis import *  # NOQA
        >>> basis = {
        >>>     'nobs1': [1, 10, 100, 10000],
        >>>     'nobs2': [1, 10, 100, 10000],
        >>> }
        >>> for params in ub.named_product(basis):
        >>>     data1 = np.random.rand(params['nobs1'])
        >>>     data2 = np.random.rand(params['nobs2'])
        >>>     data3 = np.hstack([data1, data2])
        >>>     s1 = stats_dict(data1)
        >>>     s2 = stats_dict(data2)
        >>>     s3 = stats_dict(data3)
        >>>     # Check that our combo works
        >>>     combo_s3 = combine_stats(s1, s2)
        >>>     compare = pd.DataFrame({'raw': s3, 'combo': combo_s3})
        >>>     print(compare)
        >>>     assert np.allclose(compare.raw, compare.combo)

    References:
        .. [SO7753002] https://stackoverflow.com/questions/7753002/adding-combining-standard-deviations
        .. [SO2971315] https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    """
    stats = [s1, s2]
    data = {
        "nobs": np.array([s["nobs"] for s in stats]),
        "mean": np.array([s["mean"] for s in stats]),
        "std": np.array([s["std"] for s in stats]),
        "min": np.array([s["min"] for s in stats]),
        "max": np.array([s["max"] for s in stats]),
    }
    return combine_stats_arrs(data)


def combine_stats_arrs(data):
    sizes = data["nobs"]
    means = data["mean"]
    stds = data["std"]
    mins = data["min"]
    maxs = data["max"]
    varis = stds * stds

    # TODO: ddof
    # https://github.com/Erotemic/misc/blob/28cf797b9b0f8bd82e3ebee2f6d0a688ecee2838/learn/stats.py#L128

    combo_size = sizes.sum()
    combo_mean = (sizes * means).sum() / combo_size

    mean_deltas = means - combo_mean

    sv = (sizes * varis).sum()
    sm = (sizes * (mean_deltas * mean_deltas)).sum()
    combo_vars = (sv + sm) / combo_size
    combo_std = np.sqrt(combo_vars)

    combo_stats = {
        "nobs": combo_size,
        "mean": combo_mean,
        "std": combo_std,
        "min": mins.min(),
        "max": maxs.max(),
    }
    return combo_stats
