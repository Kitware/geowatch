#!/usr/bin/env python3
"""
Group sites by "positive" / "negative" type of change and then export them as
specific regions.

Individual paths may require munging.
"""
from watch.utils import util_gis
import cmd_queue
import scriptconfig as scfg
import ubelt as ub
import watch
import json


class CroppedValidateDataset(scfg.DataConfig):
    src_dvc_dpath = 'auto'
    dst_dvc_dpath = 'auto'

    def normalize(self):
        if self.src_dvc_dpath == 'auto':
            self.src_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        if self.dst_dvc_dpath == 'auto':
            self.dst_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')


def group_site_models(config):
    data_dvc_dpath = config.src_dvc_dpath
    region_dpath = data_dvc_dpath / 'annotations/drop6/region_models'
    region_datas = list(util_gis.coerce_geojson_datas(region_dpath, format='json', workers=16))
    output_dpath = (data_dvc_dpath / 'annotations' / '_temp' / 'region_partition')

    grouped_region_fpaths = ub.ddict(list)

    # Group site summaries within regions and write them to new files.
    for info in region_datas:
        region = info['data']
        grouped_feats = ub.ddict(list)

        region_row = None

        for feat in region['features']:
            props = feat['properties']
            if props['type'] == 'region':
                assert region_row is None
                region_row = feat
            elif props['type'] in {'site_summary', ' site_summary'}:
                status = props['status'] = props['status'].strip().lower()
                props['type'] = props['type'].strip()  # fix issue
                grouped_feats[status].append(feat)
            else:
                raise KeyError

        assert region_row is not None
        for status, feats in grouped_feats.items():
            part_name = region_row['properties']['region_id'] + '_' + status
            region_part = region.copy()
            region_part.pop('features')
            region_part['name'] = part_name

            region_part['features'] = [region_row] + list(feats)

            dpath = (output_dpath / status).ensuredir()
            fpath = dpath / (part_name + '.geojson')
            fpath.write_text(json.dumps(region_part))
            grouped_region_fpaths[status].append(fpath)
    return grouped_region_fpaths


def submit_crop_jobs(config, grouped_region_fpaths):

    # In case src / dst dvc paths are different.
    # If needed set them to be the same
    src_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
    dst_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')

    output_dpath = (dst_dvc_dpath / 'Validation-V1').ensuredir()

    queue = cmd_queue.Queue.create(backend='tmux', size=8, dpath=output_dpath / '_queue')

    # TODO: could choose a region specific file more intelligently
    src_bundle_dpath = src_dvc_dpath / 'Drop6'
    src_fpath = src_bundle_dpath / 'data.kwcoco.zip'

    for status, fpaths in grouped_region_fpaths.items():
        status_dpath = (output_dpath / status).ensuredir()

        align_workers = 2
        align_aux_workers = 2

        for region_fpath in fpaths:
            # Command to crop to each site summary in the region.
            dst_dpath = (status_dpath / region_fpath.stem).ensuredir()
            command = ub.codeblock(
                rf'''
                python -m geowatch.cli.coco_align \
                    --src "{src_fpath}" \
                    --dst "{dst_dpath}" \
                    --regions "{region_fpath}" \
                    --minimum_size="128x128@10GSD" \
                    --context_factor=1 \
                    --geo_preprop=auto \
                    --force_nodata=-9999 \
                    --site_summary=True \
                    --target_gsd=5 \
                    --aux_workers={align_aux_workers} \
                    --workers={align_workers} \
                ''')
            queue.submit(command)
    queue.run()


def main():
    config = CroppedValidateDataset.cli()
    print('config = ' + ub.urepr(dict(config), nl=1))
    grouped_region_fpaths = group_site_models(config)
    submit_crop_jobs(config, grouped_region_fpaths)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/oneoffs/make_site_validation_initial_dataset.py
    """
    main()
