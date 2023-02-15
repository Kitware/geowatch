"""
Handles github actions like parameter matrices
"""
import ubelt as ub


def handle_yaml_grid(default, auto, arg):
    """
    Example:
        >>> default = {}
        >>> auto = {}
        >>> arg = ub.codeblock(
        >>>     '''
        >>>     matrix:
        >>>         foo: ['bar', 'baz']
        >>>     include:
        >>>         - {'foo': 'buz', 'bug': 'boop'}
        >>>     ''')
        >>> handle_yaml_grid(default, auto, arg)

        >>> default = {'baz': [1, 2, 3]}
        >>> arg = '''
        >>>     include:
        >>>     - {
        >>>       "thresh": 0.1,
        >>>       "morph_kernel": 3,
        >>>       "norm_ord": 1,
        >>>       "agg_fn": "probs",
        >>>       "thresh_hysteresis": "None",
        >>>       "moving_window_size": "None",
        >>>       "polygon_fn": "heatmaps_to_polys"
        >>>     }
        >>>     '''
        >>> handle_yaml_grid(default, auto, arg)
    """
    stdform_keys = {'matrix', 'include'}
    import ruamel.yaml
    print('arg = {}'.format(ub.repr2(arg, nl=1)))
    if arg:
        if arg is True:
            arg = 'auto'
        if isinstance(arg, str):
            if arg == 'auto':
                arg = auto
            if isinstance(arg, str):
                arg = ruamel.yaml.safe_load(arg)
    else:
        arg = {'matrix': default}
    if isinstance(arg, dict):
        arg = ub.udict(arg)
        if len(arg - stdform_keys) == 0 and (arg & stdform_keys):
            # Standard form
            ...
        else:
            # Transform matrix to standard form
            arg = {'matrix': arg}
    elif isinstance(arg, list):
        # Transform list form to standard form
        arg = {'include': arg}
    else:
        raise TypeError(type(arg))
    assert set(arg.keys()).issubset(stdform_keys)
    print('arg = {}'.format(ub.repr2(arg, nl=1)))
    basis = arg.get('matrix', {})
    if basis:
        grid = list(ub.named_product(basis))
    else:
        grid = []
    grid.extend(arg.get('include', []))
    return grid


def coerce_list_of_action_matrices(arg):
    """
    Preprocess the parameter grid input into a standard form
    """
    import ruamel.yaml
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()
    if isinstance(data, dict):
        pass
    action_matrices = []
    if isinstance(data, list):
        for item in data:
            action_matrices.append(item)
    elif isinstance(data, dict):
        if not len(ub.udict(data) & {'matrix', 'include'}):
            data = {'matrix': data}
        action_matrices.append(data)
    return action_matrices


def prevalidate_param_grid(arg):
    """
    Determine if something may go wrong
    """

    def validate_pathlike(p):
        if isinstance(p, str):
            p = ub.Path(p)
        else:
            p = ub.Path(p)
        if p.expand().exists():
            return True
        return False

    action_matrices = coerce_list_of_action_matrices(arg)

    # TODO: this doesn't belong in a utils folder.
    src_pathlike_keys = [
        'trk.pxl.model',
        'trk.pxl.data.test_dataset',
        'crop.src',
        'act.pxl.model',
        'act.pxl.data.test_dataset',
    ]

    logs = []

    def log_issue(k, p, msg):
        logs.append((k, p, msg))
        print(f'Key {k} with {p=} {msg}')

    for item in action_matrices:
        matrix = item['matrix']
        for k in src_pathlike_keys:
            if k in matrix:
                v = matrix[k]
                v = [v] if not ub.iterable(v) else v
                for p in v:
                    if not validate_pathlike(p):
                        log_issue(k, p, 'might not be a valid path')


def expand_param_grid(arg, max_configs=None):
    """
    Our own method for specifying many combinations. Uses the github actions
    method under the hood with our own

    Ignore:
        >>> from watch.utils.util_param_grid import *  # NOQA
        >>> arg = ub.codeblock(
            '''
            - matrix:
                trk.pxl.model: [trk_a, trk_b]
                trk.pxl.data.tta_time: [0, 4]
                trk.pxl.data.set_cover_algo: [None, approx]
                trk.pxl.data.test_dataset: [D4_S2_L8]

                act.pxl.model: [act_a, act_b]
                act.pxl.data.test_dataset: [D4_WV_PD, D4_WV]
                act.pxl.data.input_space_scale: [1GSD, 4GSD]

                trk.poly.thresh: [0.17]
                act.poly.thresh: [0.13]

                exclude:
                  #
                  # The BAS A should not run with tta
                  - trk.pxl.model: trk_a
                    trk.pxl.data.tta_time: 4
                  # The BAS B should not run without tta
                  - trk.pxl.model: trk_b
                    trk.pxl.data.tta_time: 0
                  #
                  # The SC B should not run on the PD dataset when GSD is 1
                  - act.pxl.model: act_b
                    act.pxl.data.test_dataset: D4_WV_PD
                    act.pxl.data.input_space_scale: 1GSD
                  # The SC A should not run on the WV dataset when GSD is 4
                  - act.pxl.model: act_a
                    act.pxl.data.test_dataset: D4_WV
                    act.pxl.data.input_space_scale: 4GSD
                  #
                  # The The BAS A and SC B model should not run together
                  - trk.pxl.model: trk_a
                    act.pxl.model: act_b
                  # Other misc exclusions to make the output cleaner
                  - trk.pxl.model: trk_b
                    act.pxl.data.input_space_scale: 4GSD
                  - trk.pxl.data.set_cover_algo: None
                    act.pxl.data.input_space_scale: 1GSD

                include:
                  # only try the 10GSD scale for trk model A
                  - trk.pxl.model: trk_a
                    trk.pxl.data.input_space_scale: 10GSD
            ''')
        >>> grid_items = expand_param_grid(arg)
        >>> print('grid_items = {}'.format(ub.repr2(grid_items, nl=1, sort=0)))
        >>> from watch.utils.util_dotdict import dotdict_to_nested
        >>> print(ub.repr2([dotdict_to_nested(p) for p in grid_items], nl=-3, sort=0))
        >>> print(len(grid_items))
    """
    prevalidate_param_grid(arg)
    action_matrices = coerce_list_of_action_matrices(arg)
    grid_items = []
    for item in action_matrices:
        grid_items += github_action_matrix(item)
    if max_configs is not None:
        # TODO: generator with early stop
        return grid_items[:max_configs]
    return grid_items


def github_action_matrix(arg):
    """
    Try to implement the github method. Not sure if I like it.

    References:
        https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs#expanding-or-adding-matrix-configurations

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/utils/util_param_grid.py github_action_matrix
        xdoctest -m watch.utils.util_param_grid github_action_matrix:1

    Example:
        >>> from watch.utils.util_param_grid import *  # NOQA
        >>> from watch.utils import util_param_grid
        >>> arg = ub.codeblock(
                 '''
                   matrix:
                     fruit: [apple, pear]
                     animal: [cat, dog]
                     include:
                       - color: green
                       - color: pink
                         animal: cat
                       - fruit: apple
                         shape: circle
                       - fruit: banana
                       - fruit: banana
                         animal: cat
                 ''')
        >>> grid_items = list(github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))

        grid_items = [
            {'fruit': 'apple', 'animal': 'cat', 'color': 'pink', 'shape': 'circle'},
            {'fruit': 'apple', 'animal': 'dog', 'color': 'green', 'shape': 'circle'},
            {'fruit': 'pear', 'animal': 'cat', 'color': 'pink'},
            {'fruit': 'pear', 'animal': 'dog', 'color': 'green'},
            {'fruit': 'banana'},
            {'fruit': 'banana', 'animal': 'cat'},
        ]

    Example:
        >>> from watch.utils.util_param_grid import *  # NOQA
        >>> from watch.utils import util_param_grid
        >>> arg = ub.codeblock(
                 '''
                   matrix:
                     ones: [1, 2]
                     tens: [10, 20]
                     include:
                       - ones: 1
                         color: green
                       - ones: 1
                         color: pink
                 ''')
        >>> grid_items = list(github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))


    Example:
        >>> from watch.utils.util_param_grid import *  # NOQA
        >>> arg = ub.codeblock(
                '''
                  matrix:
                    os: [macos-latest, windows-latest]
                    version: [12, 14, 16]
                    environment: [staging, production]
                    exclude:
                      - os: macos-latest
                        version: 12
                        environment: production
                      - os: windows-latest
                        version: 16
            ''')
        >>> grid_items = list(github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.repr2(grid_items, nl=1)))
        grid_items = [
            {'environment': 'staging', 'os': 'macos-latest', 'version': 12},
            {'environment': 'staging', 'os': 'macos-latest', 'version': 14},
            {'environment': 'production', 'os': 'macos-latest', 'version': 14},
            {'environment': 'staging', 'os': 'macos-latest', 'version': 16},
            {'environment': 'production', 'os': 'macos-latest', 'version': 16},
            {'environment': 'staging', 'os': 'windows-latest', 'version': 12},
            {'environment': 'production', 'os': 'windows-latest', 'version': 12},
            {'environment': 'staging', 'os': 'windows-latest', 'version': 14},
            {'environment': 'production', 'os': 'windows-latest', 'version': 14},
        ]

    Ignore:

        arg = {'matrix': {'trk.pxl.model': 'unused',
          'trk.pxl.data.test_dataset': 'unused',
          'trk.pxl.data.window_space_scale': 'unused',
          'trk.pxl.data.time_sampling': 'unused',
          'trk.pxl.data.input_space_scale': 'unused',
          'trk.poly.thresh': 'unused',
          'crop.src': 'unused',
          'crop.regions': 'truth',
          'act.pxl.data.test_dataset': ['/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Drop4-SC/data_vali_small.kwcoco.json'],
          'act.pxl.data.input_space_scale': ['8GSD'],
          'act.pxl.data.time_steps': ['auto'],
          'act.pxl.data.chip_overlap': [0.3],
          'act.poly.thresh': [0.07, 0.1, 0.13],
          'act.poly.use_viterbi': [0],
          'act.pxl.model': ['/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=1-step=23940.pt.pt',
           '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=2-step=35910-v1.pt.pt',
           '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt'],
          'include': [
               {'act.pxl.data.chip_dims': '256,256',
                'act.pxl.data.window_space_scale': '8GSD',
                'act.pxl.data.input_space_scale': '8GSD',
                'act.pxl.data.output_space_scale': '8GSD'},
               # {'act.pxl.data.chip_dims': '256,256',
               #  'act.pxl.data.window_space_scale': '4GSD',
               #  'act.pxl.data.input_space_scale': '4GSD',
               #  'act.pxl.data.output_space_scale': '4GSD'}
            ]}}

    """
    import ruamel.yaml
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()

    matrix = data.pop('matrix').copy()

    include = matrix.pop('include', data.pop('include', []))
    exclude = matrix.pop('include', data.pop('exclude', []))
    include = list(map(ub.udict, include))
    exclude = list(map(ub.udict, exclude))

    # include = [ub.udict(p) for p in matrix.pop('include', [])]
    # exclude = [ub.udict(p) for p in matrix.pop('exclude', [])]
    matrix_ = {k: (v if ub.iterable(v) else [v])
               for k, v in matrix.items()}

    def is_excluded(grid_item):
        for exclude_item in exclude:
            common1 = exclude_item & grid_item
            if common1:
                common2 = grid_item & exclude_item
                if common1 == common2 == exclude_item:
                    return True

    orig_keys = set(matrix.keys())
    include_idx_to_nvariants = ub.ddict(lambda: 0)

    def included_variants(mat_item):
        grid_item = ub.udict(mat_item)
        for include_idx, include_item in enumerate(include):
            common_orig1 = (mat_item & include_item) & orig_keys
            common_orig2 = (include_item & mat_item) & orig_keys
            if common_orig1 == common_orig2:
                include_idx_to_nvariants[include_idx] += 1
                # the key:value pairs in the object will be added to each of
                # the [original] matrix combinations if none of the key:value
                # pairs overwrite any of the original matrix values
                #
                # Note that the original matrix values will not be overwritten
                # but added matrix values can be overwritten
                grid_item = grid_item | include_item
                yield grid_item

    NEW = 1
    if NEW:
        for mat_item in map(ub.udict, ub.named_product(matrix_)):
            if not is_excluded(mat_item):
                for grid_item in included_variants(mat_item):
                    yield grid_item

        for idx, n in include_idx_to_nvariants.items():
            if n == 0:
                grid_item = include[n]
                yield grid_item
    else:
        grid_stage0 = list(map(ub.udict, ub.named_product(matrix_)))

        # Note: All include combinations are processed after exclude. This allows
        # you to use include to add back combinations that were previously
        # excluded.

        grid_stage1 = [p for p in grid_stage0 if not is_excluded(p)]

        orig_keys = set(matrix.keys())
        # Extra items are never modified by future include values include values
        # will only modify non-conflicting original grid items or create one of
        # these special immutable grid items.
        appended_items = []

        # For each object in the include list
        for include_item in include:
            ...
            any_updated = False
            for grid_item in grid_stage1:
                common_orig1 = (grid_item & include_item) & orig_keys
                common_orig2 = (include_item & grid_item) & orig_keys
                if common_orig1 == common_orig2:
                    # the key:value pairs in the object will be added to each of
                    # the [original] matrix combinations if none of the key:value
                    # pairs overwrite any of the original matrix values
                    any_updated = True
                    # Note that the original matrix values will not be overwritten
                    # but added matrix values can be overwritten
                    grid_item.update(include_item)
            if not any_updated:
                # If the object cannot be added to any of the matrix combinations, a
                # new matrix combination will be created instead.
                appended_items.append(include_item)
        grid_items = grid_stage1 + appended_items

        yield from grid_items
