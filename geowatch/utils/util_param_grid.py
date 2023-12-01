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
    print('arg = {}'.format(ub.urepr(arg, nl=1)))
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
    print('arg = {}'.format(ub.urepr(arg, nl=1)))
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
        matrix = item.get('matrix', {})
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
        >>> from geowatch.utils.util_param_grid import *  # NOQA
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
        >>> grid_items = list(expand_param_grid(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1, sort=0)))
        >>> from geowatch.utils.util_dotdict import dotdict_to_nested
        >>> print(ub.urepr([dotdict_to_nested(p) for p in grid_items], nl=-3, sort=0))
        >>> print(len(grid_items))
    """
    prevalidate_param_grid(arg)
    action_matrices = coerce_list_of_action_matrices(arg)
    num_yeilded = 0
    for item in action_matrices:
        for grid_item in extended_github_action_matrix(item):
            yield grid_item
            num_yeilded += 1
            if max_configs is not None:
                if num_yeilded >= max_configs:
                    return


def github_action_matrix(arg):
    """
    Implements the github action matrix strategy exactly as described.

    Unless I've implemented something incorrectly, I believe this method is
    limited and have extended it in :func:`extended_github_action_matrix`.

    Args:
        arg (Dict | str): a dictionary or a yaml file that resolves to a
            dictionary containing the keys "matrix", which maps parameters to a
            list of possible values. For convinieince if a single scalar value
            is detected it is converted to a list of 1 item. The matrix may
            also include an "include" and "exclude" item, which are lists of
            dictionaries that modify existing / add new matrix configurations
            or remove them. The "include" and "exclude" parameter can also be
            specified at the same level of "matrix" for convinience.

    Yields:
        item : a single entry in the grid.

    References:
        https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs#expanding-or-adding-matrix-configurations

    CommandLine:
        xdoctest -m geowatch.utils.util_param_grid github_action_matrix:2

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
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
        >>> from geowatch.utils.util_param_grid import *  # NOQA
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
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))
        grid_items = [
            {'os': 'macos-latest', 'version': 12, 'environment': 'staging'},
            {'os': 'macos-latest', 'version': 14, 'environment': 'staging'},
            {'os': 'macos-latest', 'version': 14, 'environment': 'production'},
            {'os': 'macos-latest', 'version': 16, 'environment': 'staging'},
            {'os': 'macos-latest', 'version': 16, 'environment': 'production'},
            {'os': 'windows-latest', 'version': 12, 'environment': 'staging'},
            {'os': 'windows-latest', 'version': 12, 'environment': 'production'},
            {'os': 'windows-latest', 'version': 14, 'environment': 'staging'},
            {'os': 'windows-latest', 'version': 14, 'environment': 'production'},
        ]

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
        >>> arg = ub.codeblock(
                 '''
                 matrix:
                   old_variable:
                       - null
                       - auto
                 include:
                     - old_variable: null
                       new_variable: 1
                     - old_variable: null
                       new_variable: 2
                 ''')
        >>> grid_items = list(github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))
    """
    import ruamel.yaml
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()

    matrix = data.pop('matrix', {}).copy()

    include = matrix.pop('include', data.pop('include', []))
    exclude = matrix.pop('exclude', data.pop('exclude', []))
    include = list(map(ub.udict, include))
    exclude = list(map(ub.udict, exclude))

    matrix_ = {k: (v if ub.iterable(v) else [v])
               for k, v in matrix.items()}

    orig_keys = set(matrix.keys())
    include_idx_to_nvariants = {idx: 0 for idx in range(len(include))}

    def include_modifiers(mat_item):
        """
        For each object in the include list, the key:value pairs in the object
        will be added to each of the matrix combinations if none of the
        key:value pairs overwrite any of the original matrix values. If the
        object cannot be added to any of the matrix combinations, a new matrix
        combination will be created instead. Note that the original matrix
        values will not be overwritten, but added matrix values can be
        overwritten.
        """
        grid_item = ub.udict(mat_item)
        for include_idx, include_item in enumerate(include):
            common_orig1 = (mat_item & include_item) & orig_keys
            common_orig2 = (include_item & mat_item) & orig_keys
            if common_orig1 == common_orig2:
                include_idx_to_nvariants[include_idx] += 1
                grid_item = grid_item | include_item
        return grid_item

    def is_excluded(grid_item):
        """
        An excluded configuration only has to be a partial match for it to be
        excluded. For example, the following workflow will run nine jobs: one
        job for each of the 12 configurations, minus the one excluded job that
        matches {os: macos-latest, version: 12, environment: production}, and
        the two excluded jobs that match {os: windows-latest, version: 16}.
        """
        for exclude_item in exclude:
            common1 = exclude_item & grid_item
            if common1:
                common2 = grid_item & exclude_item
                if common1 == common2 == exclude_item:
                    return True

    for mat_item in map(ub.udict, ub.named_product(matrix_)):
        grid_item = include_modifiers(mat_item)
        if not is_excluded(grid_item):
            yield grid_item

    for idx, n in include_idx_to_nvariants.items():
        if n == 0:
            grid_item = include[idx]
            yield grid_item


def extended_github_action_matrix(arg):
    """
    A variant of the github action matrix for our mlops framework that
    overcomes some of the former limitations.

    This keeps the same weird include / exclude semantics, but
    adds an additional "submatrix" component that has the following semantics.

    A submatrices is a list of dictionaries, but each dictionary may have more
    than one value, and are expanded into a list of items, similarly to a
    dictionary. In this respect the submatrix is "resolved" to a list of
    dictionary items just like "include". The difference is that when a
    common elements of a submatrix grid item matches a matrix grid item, it
    updates it with its new values and yields it immediately. Subsequent
    submatrix grid items can yield different variations of this item.
    The actions include rules are then applied on top of this.

    Args:
        arg (Dict | str): See github_action_matrix, but with new submatrices

    Yields:
        item : a single entry in the grid.

    CommandLine:
        xdoctest -m geowatch.utils.util_param_grid extended_github_action_matrix:2

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
        >>> from geowatch.utils import util_param_grid
        >>> arg = ub.codeblock(
                 '''
                   matrix:
                     fruit: [apple, pear]
                     animal: [cat, dog]
                     submatrix:
                       - color: green
                       - color: pink
                         animal: cat
                       - fruit: apple
                         shape: circle
                       - fruit: banana
                       - fruit: banana
                         animal: cat
                 ''')
        >>> grid_items = list(extended_github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
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
        >>> grid_items = list(extended_github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
        >>> from geowatch.utils import util_param_grid
        >>> # Specifying an explicit list of things to run
        >>> arg = ub.codeblock(
                 '''
                 submatrices:
                    - common_variable: a
                      old_variable: a
                    - common_variable: a
                      old_variable: null
                      new_variable: 1
                    - common_variable: a
                      old_variable: null
                      new_variable: 11
                    - common_variable: a
                      old_variable: null
                      new_variable: 2
                    - common_variable: b
                      old_variable: null
                      new_variable: 22
                 ''')
        >>> grid_items = list(extended_github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))
        >>> assert len(grid_items) == 5

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
        >>> from geowatch.utils import util_param_grid
        >>> arg = ub.codeblock(
                 '''
                 matrix:
                   common_variable:
                       - a
                       - b
                   old_variable:
                       - null
                       - auto
                 submatrices:
                     - old_variable: null
                       new_variable1:
                           - 1
                           - 2
                       new_variable2:
                           - 3
                           - 4
                     - old_variable: null
                       new_variable2:
                           - 33
                           - 44
                     # These wont be used because blag doesn't exist
                     - old_variable: blag
                       new_variable:
                           - 10
                           - 20
                 ''')
        >>> grid_items = list(extended_github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))
        >>> assert len(grid_items) == 14

    Example:
        >>> from geowatch.utils.util_param_grid import *  # NOQA
        >>> from geowatch.utils import util_param_grid
        >>> arg = ub.codeblock(
                 '''
                 matrix:
                   step1.src:
                       - dset1
                       - dset2
                       - dset3
                       - dset4
                   step1.resolution:
                       - 10
                       - 20
                       - 30
                 submatrices1:
                    - step1.resolution: 10
                      step2.resolution: [10, 15]
                    - step1.resolution: 20
                      step2.resolution: 20
                 submatrices2:
                    - step1.src: dset1
                      step2.src: big_dset1A
                    - step1.src: dset2
                      step2.src:
                         - big_dset2A
                         - big_dset2B
                    - step1.src: dset3
                      step2.src: big_dset3A
                 ''')
        >>> grid_items = list(extended_github_action_matrix(arg))
        >>> print('grid_items = {}'.format(ub.urepr(grid_items, nl=1)))
        >>> assert len(grid_items) == 20
    """
    import ruamel.yaml
    from kwutil.util_yaml import Yaml
    import os
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()

    matrix = data.pop('matrix', {}).copy()

    include = matrix.pop('include', data.pop('include', []))
    exclude = matrix.pop('exclude', data.pop('exclude', []))

    submatrices = matrix.pop('submatrices', data.pop('submatrices', []))
    submatrices = list(map(ub.udict, submatrices))

    include = list(map(ub.udict, include))
    exclude = list(map(ub.udict, exclude))

    def coerce_matrix_value(v):
        if not ub.iterable(v):
            v = [v]
        final = []
        for item in v:
            if isinstance(item, (str, os.PathLike)) and str(item).endswith(('.yaml', '.yml')):
                # use Yaml.coerce instead?
                final.extend(Yaml.load(item))
            else:
                final.append(item)
        return final

    # Special submatrices for more cartesian products, it would be good to come
    # up with a solution that does not require hard coded and a fixed number of
    # variables.
    numbered_submatrices = [
        matrix.pop('submatrices1', data.pop('submatrices1', [])),
        matrix.pop('submatrices2', data.pop('submatrices2', [])),
        matrix.pop('submatrices3', data.pop('submatrices3', [])),
        matrix.pop('submatrices4', data.pop('submatrices4', [])),
        matrix.pop('submatrices5', data.pop('submatrices5', [])),
        matrix.pop('submatrices6', data.pop('submatrices6', [])),
        matrix.pop('submatrices7', data.pop('submatrices7', [])),
        matrix.pop('submatrices8', data.pop('submatrices8', [])),
        matrix.pop('submatrices9', data.pop('submatrices9', [])),
    ]

    MULTI_SUBMATRICES = 1
    if MULTI_SUBMATRICES:
        # Try allowing for more variations. The idea is we effectively
        # want to take the cross product of multiple lists of submatrices.
        multi_submatrices = [submatrices] + numbered_submatrices
        multi_submatrices_ = []
        for submats in multi_submatrices:
            submats[:] = list(map(ub.udict, submats))
            submats_ = []
            for submatrix in submats:
                submatrix_ = {k: coerce_matrix_value(v)
                              for k, v in submatrix.items()}
                submats_.extend(list(map(ub.udict, ub.named_product(submatrix_))))
            multi_submatrices_.append(submats_)
    else:
        submatrices_ = []
        for submatrix in submatrices:
            submatrix_ = {k: coerce_matrix_value(v)
                          for k, v in submatrix.items()}
            submatrices_.extend(list(map(ub.udict, ub.named_product(submatrix_))))

    if len(data) != 0:
        raise Exception(f'Unexpected top level keys: {list(data.keys())}')

    matrix_ = {k: coerce_matrix_value(v)
               for k, v in matrix.items()}

    orig_keys = set(matrix.keys())
    include_idx_to_nvariants = {idx: 0 for idx in range(len(include))}

    def include_modifiers(mat_item):
        """
        For each object in the include list, the key:value pairs in the object
        will be added to each of the matrix combinations if none of the
        key:value pairs overwrite any of the original matrix values. If the
        object cannot be added to any of the matrix combinations, a new matrix
        combination will be created instead. Note that the original matrix
        values will not be overwritten, but added matrix values can be
        overwritten.
        """
        grid_item = ub.udict(mat_item)
        for include_idx, include_item in enumerate(include):
            common_orig1 = (mat_item & include_item) & orig_keys
            common_orig2 = (include_item & mat_item) & orig_keys
            if common_orig1 == common_orig2:
                include_idx_to_nvariants[include_idx] += 1
                grid_item = grid_item | include_item
        return grid_item

    def multisubmatrix_variants(mat_item, multi_submatrices_):
        # New version: every group of submatrices has the opportunity to
        # modify the item before yielding.
        curr_items = [mat_item]
        for submatrices_ in multi_submatrices_:
            curr_items = _submatrix_variants_loop(curr_items, submatrices_)
        yield from curr_items

    def _submatrix_variants_loop(mat_items, submatrices_):
        for item in mat_items:
            yield from submatrix_variants(item, submatrices_)

    def submatrix_variants(mat_item, submatrices_):
        """
        For each object in the include list, the key:value pairs in the object
        will be added to each of the matrix combinations if none of the
        key:value pairs overwrite any of the original matrix values. If the
        object cannot be added to any of the matrix combinations, a new matrix
        combination will be created instead. Note that the original matrix
        values will not be overwritten, but added matrix values can be
        overwritten.
        """
        grid_item = ub.udict(mat_item)
        any_modified = False
        for submat_item in submatrices_:
            common_orig1 = (mat_item & submat_item) & orig_keys
            common_orig2 = (submat_item & mat_item) & orig_keys
            if common_orig1 == common_orig2:
                grid_item = mat_item | submat_item
                yield grid_item
                any_modified = True
        if not any_modified:
            yield grid_item

    def is_excluded(grid_item):
        """
        An excluded configuration only has to be a partial match for it to be
        excluded. For example, the following workflow will run nine jobs: one
        job for each of the 12 configurations, minus the one excluded job that
        matches {os: macos-latest, version: 12, environment: production}, and
        the two excluded jobs that match {os: windows-latest, version: 16}.
        """
        for exclude_item in exclude:
            common1 = exclude_item & grid_item
            if common1:
                common2 = grid_item & exclude_item
                if common1 == common2 == exclude_item:
                    return True

    for mat_item in map(ub.udict, ub.named_product(matrix_)):
        if MULTI_SUBMATRICES:
            submat_gen = multisubmatrix_variants(mat_item, multi_submatrices_)
        else:
            submat_gen = submatrix_variants(mat_item, submatrices_)
        for item in submat_gen:
            item = include_modifiers(item)
            if not is_excluded(item):
                yield item

    for idx, n in include_idx_to_nvariants.items():
        if n == 0:
            grid_item = include[idx]
            yield grid_item
