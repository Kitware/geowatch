import argparse
import sys
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


def merge_metrics_results(region_dpaths, out_dpath=None):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            sc/ [optional]
        out_dpath: Directory to save merged results. Existing contents will
            be removed.
            Default is {common root of region_dpaths}/merged/
    '''
    if out_dpath is None:
        out_dpath = os.path.join(os.path.commonpath(region_dpaths), 'merged')
    assert out_dpath not in region_dpaths
    os.system(f'rm -r {out_dpath}')
    os.makedirs(out_dpath, exist_ok=True)

    bas_dpaths = filter(os.path.isdir,
                        (os.path.join(r, 'bas') for r in region_dpaths))
    sc_dpaths = filter(os.path.isdir,
                       (os.path.join(r, 'sc') for r in region_dpaths))
    print(bas_dpaths, sc_dpaths)

    return NotImplemented


def ensure_thumbnails(image_path, gt_dpath, region_id, coco_dset):
    '''
    Symlink and organize images in the format the metrics framework expects

    For the region visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {self.region_model.id.replace('_', '/')}/images/*/*/*.jp2")

    For the site visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {gt_ann_id.replace('_', '/')}/crops/*.tif")

    Which becomes:
    {country_code}/
        {region_num}/
            images/
                */
                    */
                        *.jp2
            {site_num}/
                crops/
                    *.tif

    Args:
        image_path: root directory to save under
        gt_dpath: $DVC_DPATH/annotations/ == smartgitlab.com/TE/annotations/
        region_id: ex. 'KR_R001'
        coco_dset: containing a video named {region_id}
    '''

    return NotImplemented


def main(args):
    parser = argparse.ArgumentParser(
        description='Score IARPA site model GeoJSON files using IARPA\'s '
        'metrics-and-test-framework')
    parser.add_argument('sites',
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
    # https://stackoverflow.com/a/49351471
    parser.add_argument('--virtualenv_cmd',
                        default='true',  # no-op command
                        nargs='+',  # hack for spaces
                        help='''
        Command to run before calling the metrics framework in a subshell.
        The metrics framework should be installed in a different virtual env
        from WATCH, using eg conda or pyenv.
        ''')
    parser.add_argument('--out_dir',
                        help='''
        Output directory where scores will be written. Each region will have
        Defaults to ./output/
        ''')

    if 0:  # TODO
        parser.add_argument('--keep_thumbnails',
                            action='store_true',
                            help='''
        Output thumbnails of region and ground truth sites to
        {out_dir}/thumbnails/
        ''')

        parser.add_argument('--merge',
                            action='store_true',
                            help='''
        Merge BAS and SC metrics from all regions and output to
        {out_dir}/merged/
        ''')
    args = parser.parse_args(args)

    # load sites
    sites = []
    for site in args.sites:
        try:
            if os.path.isfile(site):
                site = json.load(open(site))
            else:
                site = json.loads(site)
        except json.JSONDecodeError as e:  # TODO split out as decorator?
            raise json.JSONDecodeError(e.msg + ' [cut for length]',
                                       e.doc[:100] + '...',
                                       e.pos)
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
    virtualenv_cmd = ' '.join(args.virtualenv_cmd)
    ub.cmd(virtualenv_cmd, verbose=1, check=True, shell=True)

    # split sites by region
    for region_id, region_sites in ub.group_items(
            sites,
            lambda site: site['features'][0]['properties']['region_id']).items(
            ):
        with TemporaryDirectory() as site_dpath, TemporaryDirectory(
        ) as image_dpath:

            # doctor site_dpath for expected structure
            site_sub_dpath = os.path.join(site_dpath, 'latest', region_id)
            os.makedirs(site_sub_dpath, exist_ok=True)

            # copy site models to site_dpath
            for site in region_sites:
                with open(
                        os.path.join(
                            site_sub_dpath,
                            (site['features'][0]['properties']['site_id'] +
                             '.geojson')),
                        'w') as f:
                    json.dump(site, f)

            if 1:

                # link rgb images to image_dpath for viz
                img_date_dct = dict()
                for site in sites:
                    for feat in site['features'][1:]:
                        img_path = feat['properties']['source']
                        if os.path.isfile(img_path):
                            img_date_dct[img_path] = feat['properties'][
                                'observation_date']
                        else:
                            print(f'warning: image {img_path}'
                                  ' is not a valid path')
                for img_path, img_date in img_date_dct.items():
                    # use filename expected by metrics framework
                    os.symlink(
                        img_path,
                        os.path.join(
                            image_dpath, '_'.join(
                                (img_date, os.path.basename(img_path)))))

            else:  # TODO finish updating this

                ensure_thumbnails()

            # run metrics framework
            if args.out_dir is not None:
                out_dir = os.path.join(args.out_dir, region_id)
            else:
                out_dir = None
            cmd = ub.paragraph(fr'''
                {virtualenv_cmd} &&
                python {os.path.join(metrics_dpath, 'run_evaluation.py')}
                    --roi {region_id}
                    --gt_path {os.path.join(gt_dpath, 'site_models')}
                    --rm_path {os.path.join(gt_dpath, 'region_models')}
                    --sm_path {site_dpath}
                    --image_dir {image_dpath}
                    --output_dir {out_dir}
                ''')
            import subprocess
            try:
                ub.cmd(cmd, verbose=1, check=True, shell=True)
            except subprocess.CalledProcessError:
                print('error in metrics framework, probably due to zero '
                      'TP site matches.')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
