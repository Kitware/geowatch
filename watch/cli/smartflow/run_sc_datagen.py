#!/usr/bin/env python3
"""
Handles building datasets for AC/SC
"""
from watch.cli.smartflow_ingress import smartflow_ingress
from watch.cli.smartflow_egress import smartflow_egress
from watch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg


class ACSCDatasetConfig(scfg.DataConfig):
    """
    Generate cropped KWCOCO dataset for SC
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))
    output_path = scfg.Value(None, type=str, position=3, required=True, help='S3 path for output JSON')
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATD. DO NOT USE')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items

            UNUSED.
            '''))
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')
    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))

    acsc_cluster_config = scfg.Value(None, help=ub.paragraph(
        '''
        The configuration for the site-cluster-step.
        If None, then no site clustering is done.
        '''))

    acsc_align_config = scfg.Value(None, help=ub.paragraph(
        '''
        The configuration for the coco-align step
        '''))


def main():
    config = ACSCDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_generate_sc_cropped_kwcoco(config)


def run_generate_sc_cropped_kwcoco(config):
    from kwutil.util_yaml import Yaml
    from watch.utils import util_framework
    from watch.cli import coco_align
    from watch.utils import util_fsspec
    from watch.mlops.pipeline_nodes import ProcessNode

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    import watch
    print('Print current version of the code')
    ub.cmd('git log -n 1', verbose=3, cwd=ub.Path(watch.__file__).parent)

    if config.dont_recompute:
        output_path = util_fsspec.FSPath.coerce(config.output_path)
        if output_path.exists():
            # If output_path file was there, nothing to do
            return

    ingress_dir = ub.Path('/tmp/ingress')
    local_region_path = ub.Path('/tmp/region.json')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")

    # TODO:
    # Compute kwcoco-for-sc here rather than in run-bas-datagen
    ingressed_assets = smartflow_ingress(
        input_path=config.input_path,
        assets=['kwcoco_for_sc',
                'sv_out_region_models',
                'cropped_region_models_bas'],
        outdir=ingress_dir,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun,
        dont_error_on_missing_asset=True)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = download_region(config.input_region_path,
                                        local_region_path,
                                        aws_profile=config.aws_profile,
                                        strip_nonregions=True)

    # Parse region_id from original region file
    region_id = util_framework.determine_region_id(local_region_path)
    print(f'region_id={region_id}')

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    if 'sv_out_region_models' in ingressed_assets:
        input_region_path = ub.Path(ingressed_assets['sv_out_region_models']) / f'{region_id}.geojson'
    else:
        print("* Didn't find region output from SV; using region output "
              "from BAS *")
        input_region_path = ub.Path(ingressed_assets['cropped_region_models_bas']) / f'{region_id}.geojson'

    print('* Printing current directory contents (1/4)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    if config.acsc_cluster_config is not None:
        print('******************')
        print('Cluster input site summaries')
        # If specified cluster sites first.
        from watch.mlops import smart_pipeline
        site_clustering = smart_pipeline.SiteClustering(root_dpath=ingress_dir)
        acsc_cluster_config = ub.udict(Yaml.coerce(config.acsc_cluster_config))
        cluster_region_dpath = (ingress_dir / 'clustered_regions').ensuredir()
        cluster_region_fpath = cluster_region_dpath / ('clustered_' + input_region_path.name)
        tocrop_region_fpath = cluster_region_fpath
        acsc_cluster_config['src'] = input_region_path
        acsc_cluster_config['dst_dpath'] = cluster_region_dpath
        acsc_cluster_config['dst_region_fpath'] = cluster_region_fpath
        site_clustering.configure(acsc_cluster_config)
        ub.cmd(site_clustering._raw_command(), check=True, verbose=3, system=True)
    else:
        cluster_region_dpath = None
        tocrop_region_fpath = input_region_path

    print('* Printing current directory contents (2/4)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    ta1_sc_kwcoco_path = ingressed_assets['kwcoco_for_sc']

    align_config_default = ub.udict(Yaml.coerce(ub.codeblock(
        f'''
        force_nodata: -9999
        include_channels: "red|green|blue|quality"
        site_summary: True
        geo_preprop: auto
        keep: null
        convexify_regions: True
        target_gsd: 2
        context_factor: 1.5
        force_min_gsd: 4
        img_workers: {str(config.jobs)}
        aux_workers: 2
        rpc_align_method: affine_warp
        image_timeout: 20minutes
        asset_timeout: 10minutes
        verbose: 1
        ''')))
    align_config = align_config_default | Yaml.coerce(config.acsc_align_config)
    if align_config['aux_workers'] == 'auto':
        align_config['aux_workers'] = align_config['include_channels'].count('|') + 1

    # 4. Crop ingress KWCOCO dataset to region for SC
    print('******************')
    print("* Cropping KWCOCO dataset to region for SC*")
    ta1_sc_cropped_kwcoco_prefilter_path = ingress_dir / 'cropped_kwcoco_for_sc_prefilter.json'
    ta1_sc_cropped_kwcoco_path = ingress_dir / 'cropped_kwcoco_for_sc.json'

    align_config['src'] = ta1_sc_kwcoco_path
    align_config['dst'] = ta1_sc_cropped_kwcoco_prefilter_path
    align_config['regions'] = tocrop_region_fpath
    # Validate align config before running anything
    align_config = coco_align.CocoAlignGeotiffConfig(**align_config)
    print('align_config = {}'.format(ub.urepr(align_config, nl=1)))

    # Not sure if one is more stable than the other, but cmd seems fine and
    # gives us nicer logs. Prefer that one when possible.
    EXEC_MODE = 'cmd'
    if EXEC_MODE == 'import':
        coco_align.main(cmdline=False, **align_config)
    elif EXEC_MODE == 'cmd':
        align_node = ProcessNode(
            command='python -m watch.cli.coco_align',
            config=align_config,
        )
        command = align_node.final_command()
        ub.cmd(command, check=True, capture=False, verbose=3)
    else:
        raise KeyError(EXEC_MODE)

    print('* Printing current directory contents (3/4)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    ### Filter / clean geotiffs (probably should be a separate step)
    CLEAN_GEOTIFFS = 0
    if CLEAN_GEOTIFFS:
        # Detect blocky black regions in geotiffs and switch them to NODATA
        # Modifies geotiffs inplace
        remove_bad_images_node = ProcessNode(
            command='geowatch clean_geotiffs',
            in_paths={
                'src': ta1_sc_cropped_kwcoco_prefilter_path,
            },
            config={
                'prefilter_channels': 'red',
                'channels': 'red|green|blue|nir',
                'workers': 'avail',
                'dry': False,
                'probe_scale': None,
                'nodata_value': -9999,
                'min_region_size': 256,
            },
            node_dpath='.'
        )
        command = remove_bad_images_node.final_command()
        ub.cmd(command, shell=True, capture=False, verbose=3, check=True)

    REMOVE_BAD_IMAGES = 1
    if REMOVE_BAD_IMAGES:
        # Remove images that are nearly all nan
        remove_bad_images_node = ProcessNode(
            command='geowatch remove_bad_images',
            in_paths={
                'src': ta1_sc_cropped_kwcoco_prefilter_path,
            },
            out_paths={
                'dst': ta1_sc_cropped_kwcoco_path,
            },
            config={
                'channels': 'red|green|blue|pan',
                'workers': 'avail',
                'interactive': False,
                'overview': 0,
            },
            node_dpath='.'
        )
        command = remove_bad_images_node.final_command()
        ub.cmd(command, shell=True, capture=False, verbose=3, check=True)
    else:
        print('Not removing bad images')
        ta1_sc_cropped_kwcoco_prefilter_path.copy(ta1_sc_cropped_kwcoco_path)

    print('* Printing current directory contents (4/4)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    sc_cropped_assets_dir = ingress_dir / f'{region_id}'
    sc_cropped_assets_dir.ensuredir()
    # Put a dummy file in this directory so we can upload a nearly-empty folder
    # to S3
    (sc_cropped_assets_dir / 'dummy').write_text('dummy')

    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_kwcoco_for_sc'] = ta1_sc_cropped_kwcoco_path

    if cluster_region_dpath is not None:
        ingressed_assets['clustered_region_dpath'] = cluster_region_dpath

    ingressed_assets['cropped_kwcoco_for_sc_assets'] = sc_cropped_assets_dir

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=config.dryrun,
                     newline=False)


if __name__ == "__main__":
    main()
