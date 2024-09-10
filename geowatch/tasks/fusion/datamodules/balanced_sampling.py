"""
Defines two classes that help with balanced sampling (with replacement)
"""
import numpy as np
import ubelt as ub
import kwarray
import networkx as nx


try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


class BalancedSampleTree(ub.NiceRepr):
    """
    Manages a sampling from a tree of indexes. Helps with balancing
    samples over multiple criteria.

    TODO:
        Move to its own file - possibly a new module. This is a very general
        construct, and would benefit from binary-language optimizations.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.balanced_sampling import BalancedSampleTree
        >>> # Given a grid of sample locations and attribute information
        >>> # (e.g., region, category).
        >>> sample_grid = [
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'rare',       'color': "red" },
        >>>     { 'region': 'region1', 'category': 'rare',       'color': "green" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region2', 'category': 'rare',       'color': "purple" },
        >>>     { 'region': 'region2', 'category': 'rare',       'color': "blue" },
        >>> ]
        >>> #
        >>> # First we can just create a flat uniform sampling grid
        >>> # and inspect the imbalance that causes.
        >>> self = BalancedSampleTree(sample_grid, rng=0)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist0 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist0 = {}'.format(ub.urepr(hist0, nl=1)))
        >>> #
        >>> # We can subdivide the indexes based on region to improve balance.
        >>> self.subdivide('region')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist1 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist1 = {}'.format(ub.urepr(hist1, nl=1)))
        >>> #
        >>> # We can further subdivide by category.
        >>> self.subdivide('category')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist2 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist2 = {}'.format(ub.urepr(hist2, nl=1)))
        >>> #
        >>> # We can further subdivide by color, with custom weights.
        >>> weights = { 'red': .25, 'blue': .25, 'green': .4, 'purple': .1 }
        >>> self.subdivide('color', weights=weights)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist3 = ub.dict_hist([
        >>>     (g['region'], g['category'], g['color']) for g in sampled
        >>> ])
        >>> print('hist3 = {}'.format(ub.urepr(hist3, nl=1)))
        >>> hist3_color = ub.dict_hist([(g['color']) for g in sampled])
        >>> print('color weights = {}'.format(ub.urepr(weights, nl=1)))
        >>> print('hist3 (color) = {}'.format(ub.urepr(hist3_color, nl=1)))
    """
    @profile
    def __init__(self, sample_grid, rng=None):
        """
        Args:
            sample_grid (List[Dict]):
                List of items with properties to be sampled

            rng (int | None | RandomState):
                random number generator or seed
        """
        self.rng = kwarray.ensure_rng(rng)

        # validate input
        if not isinstance(sample_grid, list):
            raise TypeError(ub.paragraph(
                """
                BalancedSampleTree only accepts List[Dict], but outer type
                was {type(sample_grid)}.
                """))

        if not sample_grid:
            raise ValueError('Input sample_grid is empty')

        if not isinstance(sample_grid[0], dict):
            raise TypeError(ub.paragraph(
                """
                BalancedSampleTree only accepts List[Dict], but inner type
                was {type(sample_grid[0])}.
                """))

        self.graph = self._create_graph(sample_grid)
        self._leaf_nodes = [n for n in self.graph.nodes if self.graph.out_degree[n] == 0]

    def reseed(self, rng):
        """
        Reseed (or unseed) the random number generator

        Args:
            rng (int | None | RandomState):
                random number generator or seed
        """
        self.rng = kwarray.ensure_rng(rng)

    @profile
    def _create_graph(self, sample_grid):
        graph = nx.DiGraph()

        # make a special root node
        root_node = '__root__'
        graph.add_node(root_node, weights=None)

        for index, item in enumerate(sample_grid):
            # Using urepr in the critial loop is too slow for large sample
            # grids
            # maybe we add an option to enable this for debugging / demo?
            # label = f'{index:02d} ' + ub.urepr(item, nl=0, compact=1, nobr=1)
            label = f'{index:02d}'
            graph.add_node(index, label=label, **item)
            graph.add_edge(root_node, index)
        return graph

    @profile
    def _get_parent(self, n):
        """ Get the parent of a node (assume a tree). None if it doesnt exist """
        preds = self.graph.pred[n]
        # This function is called a lot, disable sanity checks
        # if len(preds):
        #     assert len(preds) == 1
        #     return next(iter(preds))
        # else:
        #     return None
        return next(iter(preds))

    @profile
    def _reweight(self, node, idx_child):
        if self.graph.nodes[node]['weights'] is not None:
            _weights = self.graph.nodes[node]['weights']

            # remove weight for this child
            _weights = np.delete(_weights, idx_child)

            # reweight
            if _weights.sum() != 0:
                _weights = _weights / _weights.sum()
            else:
                _weights = np.zeros(1)
            self.graph.nodes[node]['weights'] = _weights

    @profile
    def _prune_and_reweight(self, nodes):
        for parent, orphans in nodes:
            grandpa = self._get_parent(parent)
            if grandpa is None:
                # already removed this branch
                self.graph.remove_nodes_from([parent] + orphans)
                continue

            # get parent index from grandpa, remove nodes
            idx_parent = list(self.graph.successors(grandpa)).index(parent)
            self.graph.remove_nodes_from([parent] + orphans)

            # update weights of grandpa, walking up the tree
            queue = [(grandpa, idx_parent)]
            while queue:
                curr_grandpa, curr_idx_parent = queue.pop()

                num_children = len(list(self.graph.successors(curr_grandpa)))
                if num_children >= 1:
                    self._reweight(curr_grandpa, curr_idx_parent)
                else:
                    # removed only child, remove the grandparent
                    _parent = curr_grandpa
                    _grandpa = self._get_parent(curr_grandpa)
                    if _grandpa is not None:
                        _idx_parent = list(self.graph.successors(_grandpa)).index(_parent)
                        queue.append((_grandpa, _idx_parent))
                        self.graph.remove_node(curr_grandpa)

        # update leaf nodes
        self._leaf_nodes = [n for n in self._leaf_nodes if self.graph.has_node(n)]
        if len(self._leaf_nodes) == 0:
            raise ValueError("Leaf nodes became empty.")

    @profile
    def subdivide(self, key, weights=None, default_weight=0):
        """
        Adds a new layer to the tree that balances across the given attribute.

        Args:
            key (str):
                A key into the item dictionary of a sample that maps to the
                property to balance over.

            weights (None | Dict[Any, Number]):
                an optional mapping from values that ``key`` could point to
                to a numeric weight.

            default_weight (None | Number):
                if an attribute is unspecified in the weight table, this is
                the default weight it should be given. Default is 0.
        """
        remove_nodes = []
        remove_edges = []
        add_edges = []
        add_nodes = []

        # Group all leaf nodes by their direct parents

        # It is possible that we could optimize this with a column-based data
        # structure, but this current structure if far more general and easier
        # to read.
        parent_to_leafs = ub.group_items(self._leaf_nodes, key=lambda n: self._get_parent(n))

        for parent, children in parent_to_leafs.items():
            # Group children by the new attribute
            val_to_subgroup = ub.group_items(children, lambda n: self.graph.nodes[n][key])
            # try:
            #     val_to_subgroup = ub.odict(sorted(val_to_subgroup.items()))
            # except TypeError:
            #     val_to_subgroup = ub.odict(sorted(val_to_subgroup.items(), key=str))

            # Add weights to the prior parent
            if weights is not None:
                weights_group = np.asarray(list(ub.take(weights, val_to_subgroup.keys(), default=default_weight)))
                denom = weights_group.sum()
                if denom != 0:
                    weights_group = weights_group / denom
                    self.graph.nodes[parent]['weights'] = weights_group
                else:
                    # All options have zero weight, schedule group for pruning
                    remove_nodes.append((parent, children))
                    continue
            else:
                self.graph.nodes[parent]["weights"] = None

            # Create a node for each child
            for value, subgroup in val_to_subgroup.items():
                # Use a dotted name to make unambiguous tree splits
                new_parent = f'{parent}.{key}={value}'
                # Mark edges to add / remove to implement the split
                remove_edges.extend([(parent, n) for n in subgroup])
                add_edges.extend([(parent, new_parent) for n in subgroup])
                add_edges.extend([(new_parent, n) for n in subgroup])
                add_nodes.append(new_parent)

        # Modify the graph
        self.graph.remove_edges_from(remove_edges)
        self.graph.add_nodes_from(add_nodes, weights=None)
        self.graph.add_edges_from(add_edges)
        self._prune_and_reweight(remove_nodes)

    @profile
    def _sample_many(self, num):
        for _ in range(num):
            idx = self.sample()
            yield idx

    @profile
    def sample(self):
        """
        Returns:
            int
        """
        current = '__root__'
        while self.graph.out_degree(current) > 0:
            children = list(self.graph.successors(current))
            num = len(children)

            weights = self.graph.nodes[current]['weights']
            if weights is None:
                idx = self.rng.randint(0, num)
            else:
                idx = self.rng.choice(num, 1, p=weights)[0]

            current = children[idx]
        return current

    @profile
    def __len__(self):
        return len(list(self._leaf_nodes))

    @profile
    def __nice__(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        n_leafs = self.__len__()
        n_depth = len(nx.algorithms.dag.dag_longest_path(self.graph))
        return f'nodes={n_nodes}, edges={n_edges}, leafs={n_leafs}, depth={n_depth}'


class BalancedSampleForest(ub.NiceRepr):
    """
    Manages a sampling from a forest of BalancedSampleTree's. Helps with balancing
    samples in the multi-label case.

    CommandLine:
        LINE_PROFILE=1 xdoctest -m geowatch.tasks.fusion.datamodules.balanced_sampling BalancedSampleForest:1 --benchmark

    Example:
        >>> from geowatch.tasks.fusion.datamodules.balanced_sampling import BalancedSampleForest
        >>> sample_grid = [
        >>>     { 'region': 'region1', 'color': {'blue': 10, 'red': 3}},
        >>>     { 'region': 'region1', 'color': {'green': 3, 'purple': 2}},
        >>>     { 'region': 'region1', 'color': {'blue': 1}},
        >>>     { 'region': 'region1', 'color': {'green': 3, 'red': 5}},
        >>>     { 'region': 'region1', 'color': {'purple': 1, 'blue': 1}},
        >>>     { 'region': 'region2', 'color': {'blue': 5, 'red': 5}},
        >>>     { 'region': 'region2', 'color': {'green': 5, 'purple': 5}},
        >>> ]
        >>> #
        >>> self = BalancedSampleForest(sample_grid, rng=0)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist0 = ub.dict_hist([g['region'] for g in sampled])
        >>> print('hist0 = {}'.format(ub.urepr(hist0, nl=1)))
        >>> #
        >>> self.subdivide('region')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist1 = ub.dict_hist([g['region'] for g in sampled])
        >>> print('hist1 = {}'.format(ub.urepr(hist1, nl=1)))
        >>> #
        >>> self.subdivide('color')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist2 = ub.dict_hist([(g['region'],) + tuple(g['color'].keys()) for g in sampled])
        >>> print('hist2 = {}'.format(ub.urepr(hist2, nl=1)))

    Example:
        >>> # xdoctest: +REQUIRES(--benchmark)
        >>> from geowatch.tasks.fusion.datamodules.balanced_sampling import BalancedSampleForest
        >>> # Make a very large dataset to test speed constraints
        >>> sample_grid = [
        >>>     { 'region': 'region1', 'color': {'blue': 10, 'red': 3}},
        >>>     { 'region': 'region1', 'color': {'green': 3, 'purple': 2}},
        >>>     { 'region': 'region1', 'color': {'blue': 1}},
        >>>     { 'region': 'region1', 'color': {'green': 3, 'red': 5}},
        >>>     { 'region': 'region1', 'color': {'purple': 1, 'blue': 1}},
        >>>     { 'region': 'region2', 'color': {'blue': 5, 'red': 5}},
        >>>     { 'region': 'region2', 'color': {'green': 5, 'purple': 5}},
        >>> ] * 10000
        >>> #
        >>> self = BalancedSampleForest(sample_grid, rng=0)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist0 = ub.dict_hist([g['region'] for g in sampled])
        >>> print('hist0 = {}'.format(ub.urepr(hist0, nl=1)))
        >>> #
        >>> self.subdivide('region')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist1 = ub.dict_hist([g['region'] for g in sampled])
        >>> print('hist1 = {}'.format(ub.urepr(hist1, nl=1)))
        >>> #
        >>> self.subdivide('color')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist2 = ub.dict_hist([(g['region'],) + tuple(g['color'].keys()) for g in sampled])
        >>> print('hist2 = {}'.format(ub.urepr(hist2, nl=1)))

    Example:
        >>> from geowatch.tasks.fusion.datamodules.balanced_sampling import BalancedSampleForest
        >>> from collections import Counter
        >>> # Imagine you have a dataset where each sample location could have 1 or
        >>> # more category. In this case make the value a dictionary where keys
        >>> # indicate the classes in the sample and the values are the importance
        >>> # of those classes to the sample.
        >>> # Make a very large dataset to test speed constraints
        >>> sample_grid = [
        >>>     { 'image_id': 1, 'class': {'dog': 1}},  # only 1 dog in this image
        >>>     { 'image_id': 2, 'class': {'dog': 1, 'cat': 2}},  # 1 dog and 2 cats in this image
        >>>     { 'image_id': 3, 'class': {'cat': 1}},
        >>>     { 'image_id': 4, 'class': {'cat': 1}},
        >>>     { 'image_id': 5, 'class': {'cat': 1}},
        >>>     { 'image_id': 6, 'class': {'cat': 1}},
        >>>     { 'image_id': 7, 'class': {'cat': 1}},
        >>>     { 'image_id': 8, 'class': {'cat': 1}},
        >>>     { 'image_id': 9, 'class': {'cat': 1}},
        >>>     { 'image_id': 10, 'class': {'cat': 3, 'dog': 1}}, # 3 cats and 1 dog in the image
        >>>     { 'image_id': 11, 'class': {'cat': 1, 'dog': 3}}, # 3 dogs and 1 cat in the image
        >>> ]
        >>> #
        >>> self = BalancedSampleForest(sample_grid, rng=0)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> class_counts = Counter()
        >>> for sample in sampled:
        >>>     class_counts.update(sample['class'])
        >>> print('Before Balancing')
        >>> print(f'class_counts = {ub.urepr(class_counts, nl=1)}')
        >>> # Do the balance step
        >>> self.subdivide('class')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> class_counts = Counter()
        >>> for sample in sampled:
        >>>     class_counts.update(sample['class'])
        >>> print('After Balancing')
        >>> print(f'class_counts = {ub.urepr(class_counts, nl=1)}')

    TODO:
        Currently this will look at all attributes passed in each item in the
        sample grid. I think we want to specify what the attributes that could
        be balanced over are, which would help prevent a deep copy.
    """
    @profile
    def __init__(self, sample_grid, rng=None, n_trees=16, scoring='uniform'):
        """
        Args:
            sample_grid (List[Dict[str, Any]]) table of sample properties
            rng (int | None | RandomState):
                random number generator or seed
            n_trees (int): number of trees in the forest
            scoring (str): can be uniform or inverse
        """
        super().__init__()
        self.rng = rng = kwarray.ensure_rng(rng)
        self.n_trees = n_trees
        self.forest = self._create_forest(sample_grid, n_trees, scoring)

    def reseed(self, rng):
        """
        Reseed (or unseed) the random number generator

        Args:
            rng (int | None | RandomState):
                random number generator or seed
        """
        self.rng = kwarray.ensure_rng(rng)
        for tree in self.forest:
            tree.reseed(self.rng)

    @profile
    def _create_forest(self, sample_grid, n_trees, scoring):
        """
        Generate N BalancedSampleTree's, producing a hard assignment for
        each multi-label attribute. Expects a multi-label attribute to arrive
        as a dictionary with possible values as keys and frequencies as values.
        """
        import copy
        forest = []
        verbose = 1
        for idx in ub.ProgIter(range(n_trees), desc='Build balanced forests', verbose=verbose):
            local_sample_grid = copy.deepcopy(sample_grid)
            for sample in local_sample_grid:
                for key, val in sample.items():
                    if isinstance(val, dict):
                        if len(val) == 0:
                            sample[key] = None
                            continue
                        elif len(val) == 1:
                            sample[key] = list(val.keys())[0]
                            continue

                        # two or more choices
                        if scoring == 'inverse':
                            labels = list(val.keys())
                            freqs = np.asarray(list(val.values()))
                            weights = 1 - (freqs / freqs.sum())
                            weights = weights / weights.sum()
                            idx = self.rng.choice(len(labels), 1, p=weights)[0]
                            sample[key] = labels[idx]
                        elif scoring == 'uniform':
                            sample[key] = self.rng.choice(list(val.keys()))
                        else:
                            raise NotImplementedError(scoring)

            # initialize a BalancedSampleTree with this sample grid
            bst = BalancedSampleTree(local_sample_grid, rng=self.rng)
            forest.append(bst)
        return forest

    @profile
    def subdivide(self, key, weights=None, default_weight=0):
        """
        Adds a new layer to each tree in the forest that balances across the
        given attribute.

        Args:
            key (str):
                A key into the item dictionary of a sample that maps to the
                property to balance over.

            weights (None | Dict[Any, Number]):
                an optional mapping from values that ``key`` could point to
                to a numeric weight.

            default_weight (None | Number):
                if an attribute is unspecified in the weight table, this is
                the default weight it should be given. Default is 0.
        """
        for tree in self.forest:
            tree.subdivide(key, weights=weights, default_weight=default_weight)

    @profile
    def _sample_many(self, num):
        for _ in range(num):
            idx = self.sample()
            yield idx

    @profile
    def sample(self):
        """
        Uniformly sample a tree from the forest, then sample from it.

        Returns:
            int
        """
        idx = self.rng.choice(self.n_trees)
        return self.forest[idx].sample()

    @profile
    def __len__(self):
        return len(self.forest[0])

    @profile
    def __nice__(self):
        graph = self.forest[0].graph
        n_trees = self.n_trees
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_leafs = len(self)
        n_depth = len(nx.algorithms.dag.dag_longest_path(graph))
        return f'trees={n_trees}, nodes={n_nodes}, edges={n_edges}, leafs={n_leafs}, depth={n_depth}'
