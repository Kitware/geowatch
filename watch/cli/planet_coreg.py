import os
import pystac

from watch.datacube.registration.planet_to_s2 import planet_to_s2_coregister
from watch.utils.util_stac import maps


def _s2_b04_band_from_baseline_scene_dir(baseline_scene_dir):
    s2_item_id = os.path.basename(baseline_scene_dir)

    for b04_band_path in (os.path.join(baseline_scene_dir, 'B04.jp2'),
                          os.path.join(baseline_scene_dir, '{}_B04.jp2'.format(s2_item_id)),  # noqa
                          os.path.join(baseline_scene_dir, 'B04.tif'),
                          os.path.join(baseline_scene_dir, '{}_B04.tif'.format(s2_item_id)),  # noqa
                          os.path.join(baseline_scene_dir, '{}_SR_B04.tif'.format(s2_item_id))):  # noqa
        if os.path.isfile(b04_band_path):
            return b04_band_path

    raise RuntimeError("Couldn't find baseline scene B04 band file")


@maps(history_entry='coregistration')
def coreg_planet_stac_item(stac_item, outdir, baseline_scenes):
    mgrs_tile = None
    try:
        mgrs_tile = ''.join(
            map(str, (stac_item.properties["mgrs:utm_zone"],
                      stac_item.properties["mgrs:latitude_band"],
                      stac_item.properties["mgrs:grid_square"])))
    except KeyError:
        pass

    if mgrs_tile is None:
        raise RuntimeError("Couldn't parse MGRS tile for Planet "
                           "STAC Item: {}".format(stac_item.id))

    if mgrs_tile in baseline_scenes:
        baseline_scene = baseline_scenes[mgrs_tile]
    else:
        raise RuntimeError(
            "No baseline scene for MGRS tile: {}".format(mgrs_tile))

    data_asset_path = stac_item.assets['data'].href
    s2_baseline_scene_band_path = _s2_b04_band_from_baseline_scene_dir(
        baseline_scene)
    print("* Using S2 baseline scene band file "
          "(B04): {}".format(s2_baseline_scene_band_path))
    coregistered_frame = planet_to_s2_coregister(
        data_asset_path,
        s2_baseline_scene_band_path,
        outdir)

    if(coregistered_frame is not None
       and os.path.isfile(coregistered_frame)):
        # Is there some proper convention here for S2 asset names?
        stac_item.assets['data'] =\
            pystac.Asset.from_dict(
                {'href': coregistered_frame})
    else:
        # Original 'data' asset not modified at all if coregistration
        # fails, so don't update assets
        pass

    return [stac_item]
