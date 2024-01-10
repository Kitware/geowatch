from watch.mlops import smart_pipeline
import uuid
from watch.utils import util_yaml
import ubelt as ub
from rich import print

# Special keys correspond to predefined pipelines.
# Choose the pipeline we want to run.
pipeline = smart_pipeline.make_smart_pipeline('bas_building_vali')

# Visualize the pipeline on a process and IO level
pipeline.print_graphs()

if 1:
    # We can also visualize the graph for real
    import kwplot
    kwplot.autompl()

    # Change the labels a bit
    for node, data in pipeline.proc_graph.nodes(data=True):
        data['label'] = node

    for node, data in pipeline.io_graph.nodes(data=True):
        data['label'] = node

    from graphid import util
    # util.util_graphviz.show_nx(pipeline.proc_graph)
    # util.util_graphviz.show_nx(pipeline.io_graph)
    util.util_graphviz.dump_nx_ondisk(pipeline.proc_graph, 'proc_graph.png')
    util.util_graphviz.dump_nx_ondisk(pipeline.io_graph, 'io_graph.png')

config = util_yaml.Yaml.loads(ub.codeblock(
    '''
    bas_pxl.package_fpath: BAS_MODEL.pt
    sc_pxl.package_fpath: BAS_MODEL.pt

    bas_pxl.chip_dims: auto,
    bas_pxl.window_space_scale: auto,
    bas_pxl.input_space_scale: window,
    bas_pxl.output_space_scale: window,

    sc_pxl.chip_dims: auto,
    sc_pxl.window_space_scale: auto,
    sc_pxl.input_space_scale: window,
    sc_pxl.output_space_scale: window,

    bas_poly.thresh: 0.1
    sc_poly.thresh: 0.1

    bas_pxl.test_dataset: kwcoco_for_bas.kwcoco.json
    sitecrop.crop_src_fpath: VIRTUAL_KWCOCO_FOR_SC.json
    '''))

# We can overwrite performance parameters based on our needs
config.update(util_yaml.Yaml.loads(ub.codeblock(
    '''
    bas_pxl.devices: "0,"
    bas_pxl.accelerator: "gpu"
    bas_pxl.batch_size: 1

    sc_pxl.devices: "0,"
    sc_pxl.accelerator: "gpu"
    sc_pxl.batch_size: 1

    sitecrop.workers: 16
    sitecrop.aux_workers: 4

    # Hacked in
    # sitecrop.include_channels: None
    sitecrop.exclude_sensors: L8
    '''
)))

# Create a path where we will want to store everything
run_uuid = uuid.uuid4()
root_dpath = ub.Path('/dag') / ub.hash_data(str(run_uuid), base='abc')[0:8]

# Configure is the only way that the pipeline should be modified.  Use it to
# set the configuration and where the results will be stored After the pipeline
# is configured it can used directly or indirectly.
pipeline.configure(config=config, root_dpath=root_dpath, cache=False)


# Get the command to run any of the nodes you want.
# You don't have to run every node in the pipeline, but you must
# run them in topological order.

self = pipeline.nodes['sitecrop']

print('')
print(pipeline.nodes['bas_pxl'].resolved_command())
print('')
print(pipeline.nodes['bas_poly'].resolved_command())
print('')
print(pipeline.nodes['sitecrop'].resolved_command())
print('')
print(pipeline.nodes['sc_pxl'].resolved_command())
print('')
print(pipeline.nodes['sc_poly'].resolved_command())


# You don't have to worry about any of the paths except
# input paths that are external to the DAG. You can ask
# what the path will be for any artifact of the DAG.

# All of the outputs of the SC-polygon step are given in the resolved_out_paths
# dictionary
sc_out_paths = pipeline.nodes['sc_poly'].resolved_out_paths

# The final output directory that will contain the site models.
site_directory = sc_out_paths['sites_dpath']
