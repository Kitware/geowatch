

def test_variable_inputs():
    """
    Test case where a node depends on a variable length set of inputs.
    """
    from geowatch.mlops.pipeline_nodes import ProcessNode
    from geowatch.mlops.pipeline_nodes import Pipeline

    # A simple pipeline where we don't need to manage reconfiguration.
    node1 = ProcessNode(name='node1', executable='node1', out_paths={'key1': 'path1'}, node_dpath='.')
    node2 = ProcessNode(name='node2', executable='node2', out_paths={'key2': 'path2'}, node_dpath='.')
    node3 = ProcessNode(name='node3', executable='node3', out_paths={'key3': 'path3'}, node_dpath='.')

    combine_node = ProcessNode(name='combine', executable='combine', in_paths={'varpaths'})

    node1.outputs['key1'].connect(combine_node.inputs['varpaths'])
    node2.outputs['key2'].connect(combine_node.inputs['varpaths'])
    node3.outputs['key3'].connect(combine_node.inputs['varpaths'])

    dag_nodes = [
        node1,
        node2,
        node3,
        combine_node
    ]
    dag = Pipeline(dag_nodes)
    dag.print_graphs()
    dag.configure()

    print(node1.final_command())
    print(node2.final_command())
    print(node3.final_command())
    print(combine_node.final_command())

    pred_nodes = combine_node.predecessor_process_nodes()
    assert list(pred_nodes) == [node1, node2, node3]


def test_slurm_options():
    from geowatch.mlops.pipeline_nodes import ProcessNode
    from geowatch.mlops.pipeline_nodes import Pipeline

    # A simple pipeline where we don't need to manage reconfiguration.
    node1 = ProcessNode(name='node1', executable='node1.exe', out_paths={'key1': 'path1'}, node_dpath='.')
    node2 = ProcessNode(name='node2', executable='node2.exe', out_paths={'key2': 'path2'}, node_dpath='.')
    node3 = ProcessNode(name='node3', executable='node3.exe', out_paths={'key3': 'path3'}, node_dpath='.')

    combine_node = ProcessNode(name='combine', executable='combine', in_paths={'varpaths'})

    node1.outputs['key1'].connect(combine_node.inputs['varpaths'])
    node2.outputs['key2'].connect(combine_node.inputs['varpaths'])
    node3.outputs['key3'].connect(combine_node.inputs['varpaths'])

    dag_nodes = [
        node1,
        node2,
        node3,
        combine_node
    ]
    dag = Pipeline(dag_nodes)
    dag.print_graphs()

    # TODO: ensure slurm options are respected in the job itself
    dag.configure(config={
        'node2.__slurm_options__': 'foo',
        'node2.some_key': 'some_value',
    })

    import cmd_queue
    cmd_queue.Queue
    queue = cmd_queue.Queue.create(backend='slurm')
    dag.submit_jobs(queue=queue)
    print_kwargs = {
        'with_status': 0,
        'style': "colors",
        'with_locks': 0,
        # 'exclude_tags': ['boilerplate'],
    }
    queue.print_commands(**print_kwargs)

    text = queue.finalize_text()
    print(text)

    new_order = queue.order_jobs()
    for job in new_order:
        print(job.__dict__)
    job = new_order[0]
