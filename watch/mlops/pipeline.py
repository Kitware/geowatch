import networkx as nx
import itertools as it
import ubelt as ub


def _iter_paths(paths):
    import os
    # Iterate over output paths sequentially
    if isinstance(paths, (str, os.PathLike)):
        yield paths
    else:
        if isinstance(paths, dict):
            for v in paths.values():
                if isinstance(v, list):
                    yield from v
                else:
                    yield v
        elif isinstance(paths, list):
            yield from paths


def _iter_deps(depends):
    if depends:
        if isinstance(depends, list):
            yield from depends
        else:
            yield depends


class Step:
    """
    Represents a single pipeline step

    Example:
        from watch.mlops.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.submit(
            name='foo',
            command='read in, write out',
            in_paths=['in'],
            out_paths=['out']
        )


    """
    def __init__(step, name, command, in_paths, out_paths, resources=None,
                 depends=None, enabled=True, stage=None):
        """
        Args:

        """
        if resources is None:
            resources = {}
        step.name = name
        step._command = command

        if isinstance(in_paths, list):
            in_paths = dict(enumerate(in_paths))
        if isinstance(out_paths, list):
            out_paths = dict(enumerate(out_paths))

        step.in_paths = ub.udict(in_paths)
        step.out_paths = ub.udict(out_paths)
        step.resources = ub.udict(resources)
        step.depends = depends
        step.stage = stage
        #
        # Set later
        step.enabled = enabled
        step.will_exist = None
        step.otf_cache = True  # on-the-fly cache checking

    @property
    def node_id(step):
        """
        The experiment manager constructs output paths such that they
        are unique given the specific set of inputs and parameters. Thus
        the output paths are sufficient to determine a unique id per step.
        """
        return step.name + '_' + ub.hash_data(step.out_paths)[0:12]

    @property
    def command(step):
        if step.otf_cache and step.enabled != 'redo':
            return step.test_is_computed_command() + ' || ' + step._command
        else:
            return step._command

    def test_is_computed_command(step):
        test_expr = ' -a '.join(
            [f'-e "{p}"' for p in step.out_paths.values()])
        test_cmd = 'test ' +  test_expr
        return test_cmd

    def _iter_opaths(self):
        return _iter_paths(self.out_paths)

    def _iter_ipaths(self):
        return _iter_paths(self.in_paths)

    @ub.memoize_property
    def does_exist(self):
        # return all(self.out_paths.map_values(lambda p: p.exists()).values())
        return all(ub.Path(p).expand().exists() for p in self._iter_opaths())


class Pipeline:

    def __init__(self):
        self.steps = []

    def submit(self, **kwargs):
        step = Step(**kwargs)
        self.steps.append(step)
        return step

    def _populate_explicit_dependency_queue(self, queue):
        for step in self.steps:
            qdepends = [d._qjob for d in _iter_deps(step.depends)]
            step._qjob = queue.submit(
                command=step.command,
                name=step.name,
                depends=qdepends,
            )

    def _populate_implicit_dependency_queue(self, queue, skip_existing=False):
        g = self.connect_steps(skip_existing=skip_existing)

        common_submitkw = {}

        # queue = cmd_queue.Queue.create(
        #     backend=config['backend'], name='prep-ta2-dataset', size=1, gres=None,
        #     environ=environ)

        # Submit steps to the scheduling queue
        for node in g.graph['order']:
            # Skip duplicate jobs
            step = g.nodes[node]['step']
            if step.node_id in queue.named_jobs:
                continue
            depends = []
            for other, _ in list(g.in_edges(node)):
                dep_step = g.nodes[other]['step']
                if dep_step.enabled:
                    depends.append(dep_step.node_id)

            if step.will_exist and step.enabled:
                step._qjob = queue.submit(
                    command=step.command, name=step.node_id, depends=depends,
                    **common_submitkw)

    def _update_stage_otf_cache(self, cache):
        """
        We have annotated some steps with stages, build a metagraph based on
        that.

        Args:
            cache (List[str] | bool):
                the list of stages to use the otf_cache on. If True then
                all stages are assumed to be cached. Otherwise any ancestor
                of a cached stage also has its cache enabled.
        """
        g = self._build_step_graph()
        stages = self._build_stage_graph()

        if isinstance(cache, int):
            cache = list(stages.nodes) if cache else []

        # For each requested cached stages, also enable the cache of its
        # ancestors. I.e. for any item where the cache is explicitly enabled,
        # enable all of its dependencies (ancestors) as well.
        for stage_name in cache:
            stages.nodes[stage_name]['cache'] = 1
            for n in nx.ancestors(stages, stage_name):
                stages.nodes[n]['cache'] = 1

        for stage_name, stage_data in stages.nodes(data=True):
            stage_data['label'] = '{stage}:cache={cache}'.format(**stage_data)

        # Update the steps themselves
        for stage_name, stage_data in stages.nodes(data=True):
            for step_name in stage_data['members']:
                g.nodes[step_name]['step'].otf_cache = stage_data['cache']

        if 1:
            from cmd_queue.util.util_networkx import write_network_text
            print('Stage Graph:')
            write_network_text(stages)

    def _build_stage_graph(self):
        g = self._build_step_graph()

        stage_to_steps = ub.ddict(list)
        for step in self.steps:
            stage_to_steps[step.stage].append(step.name)

        # Compute the condensed stage graph
        sccs = list(stage_to_steps.values())
        stages = nx.condensation(g, sccs)
        mapping = {}
        for n, stage_data in stages.nodes(data=True):
            stage = g.nodes[stage_data['members'][0]]['step'].stage
            mapping[n] = stage
            stage_data['stage'] = stage
            stage_data['cache'] = 0
        stages = nx.relabel_nodes(stages, mapping)
        return stages

    def _build_step_graph(self):
        steps = self.steps
        g = nx.DiGraph()
        outputs_to_step = ub.ddict(list)
        inputs_to_step = ub.ddict(list)
        for step in steps:
            for path in step._iter_ipaths():
                inputs_to_step[path].append(step.name)
            for path in step._iter_opaths():
                outputs_to_step[path].append(step.name)
            g.add_node(step.name, step=step)

        inputs_to_step = ub.udict(inputs_to_step)
        outputs_to_step = ub.udict(outputs_to_step)

        common = list((inputs_to_step & outputs_to_step).keys())
        for path in common:
            isteps = inputs_to_step[path]
            osteps = outputs_to_step[path]
            for istep, ostep in it.product(isteps, osteps):
                g.add_edge(ostep, istep)

        #
        # Determine which steps are enabled / disabled
        sorted_nodes = list(nx.topological_sort(g))
        g.graph['order'] = sorted_nodes
        return g

    def connect_steps(self, skip_existing=False):
        """
        Build the graph that represents this pipeline using the inputs /
        outputs to find the dependencies
        """
        # Determine the interaction / dependencies between step inputs /
        # outputs
        g = self._build_step_graph()

        for node in g.graph['order']:
            step = g.nodes[node]['step']
            # if config['skip_existing']:
            ancestors_will_exist = all(
                g.nodes[ancestor]['step'].will_exist
                for ancestor in nx.ancestors(g, step.name)
            )
            if skip_existing and step.enabled != 'redo' and step.does_exist:
                step.enabled = False
            step.will_exist = (
                (step.enabled and ancestors_will_exist) or
                step.does_exist
            )

        if 0:
            from cmd_queue.util.util_networkx import write_network_text
            write_network_text(g)
        return g
