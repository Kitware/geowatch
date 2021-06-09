import datetime
import logging
from pathlib import Path

import click
import kwcoco
import kwimage
import ndsampler
from tqdm import tqdm

from . import detector
from .tables import facc_description
from .utils import setup_logging

log = logging.getLogger(__name__)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
def predict(dataset, deployed, output):
    dset = kwcoco.CocoDataset(dataset)
    dataset = Path(dataset)

    default_output_filename = 'out_{}.kwcoco.json'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output is None:
        output = Path('/tmp', default_output_filename)
    else:
        output = Path(output)
        if output.is_dir():
            output = output.joinpath(default_output_filename)

    log.debug('dset {}'.format(dset))
    log.debug('weights: {}'.format(deployed))
    log.debug('output: {}'.format(output))

    output_dset = kwcoco.CocoDataset()
    for cid, cat in enumerate(detector.feature_mapping[1:], start=0):
        output_dset.add_category(
            cat, supercategory='landcover', id=cid,
            description=facc_description[cat],
            color=detector.cmap8[cid + 1].tolist()
        )

    log.debug('output {}'.format(output_dset))
    # log.debug('dset {}'.format(pformat(output_dset.cats)))

    # landcover detector is currently trained on 8-band WV images
    gids = [img['id'] for img in dset.imgs.values() if img['num_bands'] == 8]

    log.info('Found {} usable images out of {}'.format(len(gids), len(dset.imgs)))

    sampler = ndsampler.CocoSampler(dset)

    model = detector.load_model(deployed, num_outputs=15, num_channels=8)

    for gid in tqdm(gids):
        img_info = dset.imgs[gid]

        filename = dataset.parent.joinpath(img_info['file_name'])
        try:
            # remove video info
            for key in ('video_id', 'warp_img_to_vid', 'frame_index'):
                img_info.pop(key, None)

            output_dset.add_image(**img_info)
            # log.info('loading file: {}  {}'.format(gid, filename.name))
            img = sampler.load_image(gid, cache=False)

            features = detector.run(model, img, img_info)

            for shp, cid in features:
                poly = kwimage.Polygon.from_shapely(shp)
                box = kwimage.structs.Boxes(shp.bounds, 'ltrb')
                output_dset.add_annotation(
                    gid, cid,
                    bbox=box,
                    segmentation=poly.to_coco('new')
                )
        except KeyboardInterrupt:
            log.info('interrupted')
            break
        except Exception as e:
            log.exception('Unable to load {}'.format(filename))

    output_dset.dump(str(output), indent=2)
    log.info('output written to {}'.format(output))


if __name__ == '__main__':
    setup_logging()
    predict()
