import argparse
import logging
import json
import os
import sys
import copy
from contextlib import contextmanager
import shutil

from shapely.geometry import shape, mapping


_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate regions and sites across "
                    "pseudo-live iterations")

    parser.add_argument('region_id',
                        type=str,
                        help="Region ID")
    parser.add_argument("--input_region_path",
                        type=str,
                        required=True,
                        help="Path to input region file for current interval")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        required=True,
                        help="Output dir for consolidated regions / sites")
    parser.add_argument("-s", "--performer-suffix",
                        type=str,
                        default=None,
                        help="Performer suffix if present, e.g. KIT")
    parser.add_argument("-i", "--iou-threshold",
                        type=float,
                        default=0.5,
                        help="IOU Threshold for determining duplicates"
                             "(default: 0.5)")
    parser.add_argument('ta2_collated_dir_previous',
                        type=str,
                        help="Path to TA2 collated output dir for "
                             "previous iteration")
    parser.add_argument('ta2_collated_dir_current',
                        type=str,
                        help="Path to TA2 collated output dir for "
                             "current iteration")
    parser.add_argument('--just-deconflict',
                        action='store_true',
                        default=False,
                        help="Don't copy previous sites, just deconflict "
                             "current site IDs with respect to previous")

    pseudolive_consolidate(**vars(parser.parse_args()))


def reindex_ids(site_or_region):
    # Reindex "id" values (not "site_id") to ensure there are no
    # duplicates; modifies in place.  Only needed for sites or region
    # files that we merge / modify
    for new_id, feat in enumerate(site_or_region.get('features', ())):
        feat['id'] = str(new_id)

    return site_or_region


@contextmanager
def _yield_first_feature(site_or_region, type_):
    # Assumes there's only a single 'site' feature
    for feat in site_or_region.get('features', ()):
        if feat['properties']['type'] == type_:
            yield feat
            return


def _load_region_data(region_id, ta2_collated_dir, performer_suffix=None):
    if performer_suffix is not None:
        region_path = os.path.join(
            ta2_collated_dir, 'region_models',
            "{}_{}.geojson".format(region_id, performer_suffix))
    else:
        region_path = os.path.join(
            ta2_collated_dir, 'region_models',
            "{}.geojson".format(region_id))

    if os.path.isfile(region_path):
        with open(region_path) as f:
            return json.load(f)
    else:
        return None


def _load_site_data(site_id, ta2_collated_dir, performer_suffix=None):
    if performer_suffix is not None:
        site_path = os.path.join(
            ta2_collated_dir, 'site_models',
            "{}_{}.geojson".format(site_id, performer_suffix))
    else:
        site_path = os.path.join(
            ta2_collated_dir, 'site_models',
            "{}.geojson".format(site_id))

    with open(site_path) as f:
        return json.load(f)


def pseudolive_consolidate(region_id,
                           input_region_path,
                           ta2_collated_dir_previous,
                           ta2_collated_dir_current,
                           outdir,
                           iou_threshold,
                           performer_suffix=None,
                           just_deconflict=False):
    with open(input_region_path) as f:
        input_region = json.load(f)

    with _yield_first_feature(input_region, type_='region') as ir:
        new_region_end_date = ir['properties']['end_date']

    os.makedirs(os.path.join(outdir, 'region_models'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'site_models'), exist_ok=True)

    previous_region = _load_region_data(
        region_id, ta2_collated_dir_previous, performer_suffix)
    previous_sites_by_id = {f['properties']['site_id']: f
                            for f in previous_region.get('features', ())
                            if f['properties']['type'] == 'site_summary'}

    current_region = _load_region_data(
        region_id, ta2_collated_dir_current, performer_suffix)
    if current_region is None:
        current_sites_by_id = {}
    else:
        current_sites_by_id = {f['properties']['site_id']: f
                               for f in current_region.get('features', ())
                               if f['properties']['type'] == 'site_summary'}

    overlaps = []
    for pss in previous_sites_by_id.values():
        psid = pss['properties']['site_id']
        previous_sites_by_id[psid] = pss

        for css in current_sites_by_id.values():
            csid = css['properties']['site_id']
            current_sites_by_id[csid] = css

            iou = compute_iou(pss, css)
            if iou >= iou_threshold:
                overlaps.append((iou, psid, csid))

    sorted_overlaps = sorted(overlaps, key=lambda o: o[0], reverse=True)

    output_region = copy.deepcopy(previous_region)
    # Strip out site summaries from previous region to create initial
    # new region
    output_region['features'] = [f for f in output_region.get('features', ())
                                 if f['properties']['type'] != 'site_summary']
    with _yield_first_feature(output_region, type_='region') as outr:
        outr['properties']['end_date'] = new_region_end_date

    # TODO: Move this function to top-level scope
    def _combine_sites(psid, csid):
        pss = previous_sites_by_id[psid]
        css = current_sites_by_id[csid]

        new_site_summary = copy.deepcopy(pss)
        new_site_summary['properties']['end_date'] =\
            css['properties']['end_date']

        new_site_geometry =\
            shape(pss['geometry']).union(shape(css['geometry'])).buffer(0)
        new_site_summary['geometry'] = mapping(new_site_geometry)

        # Merge sites
        prev_site = _load_site_data(
            psid, ta2_collated_dir_previous, performer_suffix)
        curr_site = _load_site_data(
            csid, ta2_collated_dir_current, performer_suffix)

        new_site = copy.deepcopy(prev_site)
        # Set the end date of the previous (to be re-used site to the
        # end of the current matching site)
        with _yield_first_feature(new_site, type_='site') as nssf:
            with _yield_first_feature(curr_site, type_='site') as cssf:
                nssf['properties']['end_date'] =\
                    cssf['properties']['end_date']

            nssf['geometry'] = mapping(new_site_geometry)

        # TODO: Update "phase_transition_days" of last site "observation"?
        new_site['features'].extend(
            [s for s in curr_site.get('features', ())
             if s['properties']['type'] == 'observation'])

        reindex_ids(new_site)

        return new_site_summary, new_site

    # TODO: Move this function to top-level scope
    def _deconflict_new_site(psid, csid):
        # Just copy new site over and re-use previous ID
        css = current_sites_by_id[csid]

        # Use current instead of previous as starting point
        new_site_summary = copy.deepcopy(css)
        new_site_summary['properties']['site_id'] = psid

        curr_site = _load_site_data(
            csid, ta2_collated_dir_current, performer_suffix)

        # Again starting with current instead of previous site
        new_site = copy.deepcopy(curr_site)
        # Set the end date of the previous (to be re-used site to the
        # end of the current matching site)
        with _yield_first_feature(new_site, type_='site') as nssf:
            nssf['properties']['site_id'] = psid

        return new_site_summary, new_site

    matched_previous_sites = set()
    matched_current_sites = set()
    for iou, psid, csid in sorted_overlaps:
        # Only doing one-to-one site matching for now
        if(psid in matched_previous_sites  # noqa: E275
           or csid in matched_current_sites):
            continue

        print("Matched site {} (previous) to {} (current)"
              " with {:.2f} IOU".format(psid, csid, iou))

        matched_previous_sites.add(psid)
        matched_current_sites.add(csid)

        if just_deconflict:
            new_site_summary, new_site = _deconflict_new_site(psid, csid)
        else:
            new_site_summary, new_site = _combine_sites(psid, csid)

        output_region['features'].append(new_site_summary)

        if performer_suffix is None:
            output_path = os.path.join(
                outdir, 'site_models', "{}.geojson".format(psid))
        else:
            output_path = os.path.join(
                outdir, 'site_models',
                "{}_{}.geojson".format(psid, performer_suffix))

        with open(output_path, 'w') as f:
            json.dump(new_site, f, indent=2)

    # Keep track of highest id used for previous sites for when we
    # re-issue IDs for current sites later in this function; starting
    # at -1 as the first site we want to issue should have an ID value
    # of 0
    highest_psid_value = -1
    for psid, ps in previous_sites_by_id.items():
        _, _, psid_value = psid.split('_')
        psid_value = int(psid_value)
        if psid_value > highest_psid_value:
            highest_psid_value = psid_value

        if(psid in matched_previous_sites  # noqa: E275
           or just_deconflict):
            # Already added during merging; or just deconflicting current sites
            continue
        else:
            output_region['features'].append(ps)

            if performer_suffix is not None:
                site_src_path = os.path.join(
                    ta2_collated_dir_previous, 'site_models',
                    "{}_{}.geojson".format(psid, performer_suffix))
                site_dst_path = os.path.join(
                    outdir, 'site_models',
                    "{}_{}.geojson".format(psid, performer_suffix))
            else:
                site_src_path = os.path.join(
                    ta2_collated_dir_previous, 'site_models',
                    "{}.geojson".format(psid))
                site_dst_path = os.path.join(
                    outdir, 'site_models',
                    "{}.geojson".format(psid))

            shutil.copy(site_src_path, site_dst_path)

            print("Copied previous site: {}".format(psid))

    # Re-issue site_ids for "current" sites (that were not merged)
    next_site_id_value = highest_psid_value + 1
    for csid, cs in current_sites_by_id.items():
        if csid in matched_current_sites:
            continue
        else:
            updated_site_id = "{}_{:0>4}".format(region_id, next_site_id_value)
            next_site_id_value += 1
            cs['properties']['site_id'] = updated_site_id

            output_region['features'].append(cs)

            curr_site_data = _load_site_data(
                csid, ta2_collated_dir_current, performer_suffix)

            with _yield_first_feature(curr_site_data, type_='site') as cssf:
                cssf['properties']['site_id'] = updated_site_id

            if performer_suffix is not None:
                site_dst_path = os.path.join(
                    outdir, 'site_models',
                    "{}_{}.geojson".format(updated_site_id, performer_suffix))
            else:
                site_dst_path = os.path.join(
                    outdir, 'site_models',
                    "{}.geojson".format(updated_site_id))

            with open(site_dst_path, 'w') as f:
                json.dump(curr_site_data, f, indent=2)

            print("Copied current site: {} and renamed to: {}".format(
                csid, updated_site_id))

    if performer_suffix is not None:
        region_dst_path = os.path.join(
            outdir, 'region_models',
            "{}_{}.geojson".format(region_id, performer_suffix))
    else:
        region_dst_path = os.path.join(
            outdir, 'region_models',
            "{}.geojson".format(region_id))

    reindex_ids(output_region)
    with open(region_dst_path, 'w') as f:
        json.dump(output_region, f, indent=2)

    print("Wrote consolidated output files to: {}".format(outdir))


def compute_iou(site_summary_1, site_summary_2):
    """
    Addapted from MITRE Corp's Smart Nifi implementation
    https://smartgitlab.com/infrastructure/smart-nifi/-/blob/main/python_scripts/smart_nifi/consolidate_split.py

    Compare two sites and determine if they represent "roughly" the same area.

    this IS going to fail under the following circumstances:
        1) polar regions
        2) sites that cross the 180DEG line

    :param shape1: A Site that is trusted
    :type shape1: a site_summary geojson
    :param shape2: A Site that may be a duplicate
    :type shape2: a site_summary geojson
    :return: iou
    :rtype: float
    """

    # first let's do BBOX checks, because those are incredibly easy, and will
    # rule out 95% of the cases
    shape1 = shape(site_summary_1["geometry"])
    shape2 = shape(site_summary_2["geometry"])

    if shape1.bounds[0] > shape2.bounds[2]:
        _logger.debug(
            "Site %s is strictly east of %s",
            site_summary_2["properties"]["site_id"],
            site_summary_1["properties"]["site_id"],
        )
        return 0.0
    if shape1.bounds[2] < shape2.bounds[0]:
        _logger.debug(
            "Site %s is strictly west of %s",
            site_summary_2["properties"]["site_id"],
            site_summary_1["properties"]["site_id"],
        )
        return 0.0
    if shape1.bounds[1] > shape2.bounds[3]:
        _logger.debug(
            "Site %s is strictly south of %s",
            site_summary_2["properties"]["site_id"],
            site_summary_1["properties"]["site_id"],
        )
        return 0.0
    if shape1.bounds[3] < shape2.bounds[1]:
        _logger.debug(
            "Site %s is strictly north of %s",
            site_summary_2["properties"]["site_id"],
            site_summary_1["properties"]["site_id"],
        )
        return 0.0

    # now that we've filtered most of the cases out, let's do the hard work
    # of checking polygon intersections.

    intersection_area = shape1.intersection(shape2).area
    # probably rare, but worth a check!
    if intersection_area == 0:
        _logger.debug(
            "Site %s does not intersect %s",
            site_summary_2["properties"]["site_id"],
            site_summary_1["properties"]["site_id"],
        )
        return 0.0

    union_area = shape1.union(shape2).area
    if union_area == 0.0:
        return 0.0

    iou = intersection_area / union_area
    _logger.debug(
        "IOU of %s:%s is %0.4f",
        site_summary_2["properties"]["site_id"],
        site_summary_1["properties"]["site_id"],
        iou
    )
    return iou


if __name__ == "__main__":
    sys.exit(main())
