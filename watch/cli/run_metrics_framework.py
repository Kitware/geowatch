import argparse
import os
import json
from tempfile import TemporaryDirectory
import ubelt as ub
'''
REGIONS_VALI=(
    "KR_R001"
    "KR_R002"
)
REGIONS_TRAIN=(
    "US_R001"
    "BR_R001"
    "BR_R002"
    "LT_R001"
    "BH_R001"
    "NZ_R001"
)
'''


def merge_metrics_results():
    return NotImplemented


def main(args):
    parser = argparse.ArgumentParser(
        description='Score IARPA site model GeoJSON files using IARPA\'s '
        'metrics-and-test-framework')
    parser.add_argument('--sites',
                        required=True,
                        nargs='+',
                        help='''
        List of paths or serialized JSON strings containg v2 site models.
        All region_ids from these sites will be scored, and it will be assumed
        that there are no other sites in these regions.
        ''')
    parser.add_argument('--gt_dpath',
                        help='''
        Path to a local copy of the ground truth annotations,
        https://smartgitlab.com/TE/annotations.
        If None, use the environment variable DVC_DPATH to find
        $DVC_DPATH/annotations.
        ''')
    parser.add_argument('--metrics_dpath',
                        help='''
        Path to a local copy of the metrics framework,
        https://smartgitlab.com/TE/metrics-and-test-framework.
        If None, use the environment variable METRICS_DPATH.
        ''')
    parser.add_argument('--virtualenv_cmd',
                        default='conda activate te',
                        help='''
        Command to run before calling the metrics framework in a subshell. 
        The metrics framework should be installed in a different virtual env
        from WATCH, using eg conda or pyenv.
        ''')
    parser.add_argument('--out_dir',
                        help='''
        Output directory where scores will be written.
        Defaults to ./output/{region_id}/
        ''')
    args = parser.parse_args(args)

    # load sites
    sites = []
    for site in args.sites:
        if os.path.isfile(site):
            site = json.load(open(site))
        else:
            site = json.loads(site)
        sites.append(site)

    # normalize paths
    if args.gt_dpath is not None:
        gt_dpath = os.path.abspath(args.gt_dpath)
    else:
        gt_dpath = os.path.join(os.environ['DVC_DPATH'], 'annotations')
    assert os.path.isdir(gt_dpath), gt_dpath
    if args.metrics_dpath is not None:
        metrics_dpath = os.path.abspath(args.metrics_dpath)
    else:
        metrics_dpath = os.environ['METRICS_DPATH']
    assert os.path.isdir(metrics_dpath), metrics_dpath
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    # validate virtualenv command
    ub.cmd(args.virtualenv_cmd, check=True)

    # split sites by region
    for region_id, region_sites in ub.group_items(
            sites,
            lambda site: site['features'][0]['properties']['region_id']).items(
            ):
        with TemporaryDirectory() as site_dpath, TemporaryDirectory(
        ) as image_dpath:

            # copy site models to site_dpath
            for site in region_sites:
                with open(
                        os.path.join(
                            site_dpath,
                            site['features'][0]['properties']['site_id']),
                        'w') as f:
                    json.dump(site, f)

            # link rgb images to image_dpath for viz
            img_date_dct = dict()
            for site in sites:
                for feat in site['features'][1:]:
                    img_path = feat['properties']['source']
                    if os.path.isfile(img_path):
                        img_date_dct[img_path] = feat['properties'][
                            'observation_date']
                    else:
                        print(f'warning: image {img_path} is not a valid path')
            for img_path, img_date in img_date_dct.items():
                # use filename expected by metrics framework
                os.symlink(
                    img_path,
                    os.path.join(
                        image_dpath, '_'.join(img_date,
                                              os.path.basename(img_path))))

            # run metrics framework
            cmd = ub.paragraph(f'''
                {args.virtualenv_cmd};
                python {os.path.join(metrics_dpath, 'run_evaluation.py')}
                    --roi {region_id}
                    --gt_path {os.path.join(gt_dpath, 'site_models')}
                    --rm_path {os.path.join(gt_dpath, 'region_models', region_id + .geojson)}
                    --sm_path {site_dpath}
                    --image_path {image_dpath}
                    --output_dir {args.out_dir}
                ''')
            ub.cmd(cmd, shell=False, verbose=0, check=True)
