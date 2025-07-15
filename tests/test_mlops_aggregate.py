
def test_aggregate_with_unhashable_columns():
    from geowatch.mlops.aggregate import Aggregator
    agg = Aggregator.demo(rng=0, num=100, include_unhashable=True)
    agg.build()
    assert isinstance(agg.params['params.demo_node.unhashable_list'].iloc[0], list), 'The raw params should remember the object form if possible'
    assert isinstance(agg.effective_params['params.demo_node.unhashable_list'].iloc[0], str), 'The effective params should all be hashable'
    assert isinstance(agg.params['params.demo_node.unhashable_dict'].iloc[0], dict), 'The raw params should remember the object form if possible'
    assert isinstance(agg.effective_params['params.demo_node.unhashable_dict'].iloc[0], str), 'The effective params should all be hashable'
