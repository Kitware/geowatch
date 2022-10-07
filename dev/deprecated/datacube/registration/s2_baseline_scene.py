import os
import sys
import glob
import xml.etree.ElementTree as ET
from osgeo import gdal
import numpy as np
import pandas as pd
import pystac


def print_usage():
    print("Usage: python s2_baseline_scene.py input_folder_safe")
    return


def safedir_to_xml(safedir):
    '''
    Grab the metadata file needed for find_baseline_scene, assuming original Sentinel-2 .SAFE directory structure
    '''
    xmls = glob.glob(os.path.join(safedir, 'GRANULE', '*', 'MTD_TL.xml'),
                     recursive=True)
    assert len(xmls) == 1
    return xmls[0]


def find_baseline_scene(xmls, return_paths=False):
    '''
    Choose a Sentinel-2 L1C scene to serve as a reference for coregistration.

    Args:
        xmls: list of 'MTD_TL.xml' files corresponding to scenes
        return_paths: values of return dict are paths to baseline granuledirs instead

    Returns:
        Dict[mgrs_tile: str, pd.DataFrame]:
            for each MGRS tile represented in xmls,
            the granuledir of the chosen scene and its scores on quality heuristics

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # xdoctest: +REQUIRES(--network, --ta1)
        >>> from watch.datacube.registration.s2_baseline_scene import *
        >>> from watch.demo.sentinel2_demodata import grab_sentinel2_product
        >>> #
        >>> safedirs = [str(grab_sentinel2_product(i).path) for i in range(3)]
        >>> input_dirs = []
        >>> # HACK: Re-arranging input "granuledirs" to reflect what
        >>> # we receive from the Atmospheric Correction algorithm
        >>> for safedir in safedirs:
        >>>     col, _, datetime, _, _, mgrs, _ = os.path.basename(safedir).split('_')
        >>>     input_dir = os.path.join(os.path.dirname(safedir), '_'.join((col, 'MSI', 'L2A', mgrs, datetime, datetime)))
        >>>     input_image_data_dir = os.path.join(input_dir, 'IMG_DATA')
        >>>     os.makedirs(input_image_data_dir, exist_ok=True)
        >>>     mtd_tl_xml_path = safedir_to_xml(safedir)
        >>>     if not os.path.isfile(os.path.join(input_dir, 'MTD_TL.xml')):
        >>>         os.symlink(mtd_tl_xml_path, os.path.join(input_dir, 'MTD_TL.xml'))
        >>>     for b04 in glob.glob(os.path.join(os.path.dirname(safedir_to_xml(safedir)), 'IMG_DATA', '*_B04.jp2')):
        >>>         outpath = os.path.join(input_dir, os.path.basename(b04).replace(".jp2", ".tif"))
        >>>         if not os.path.isfile(outpath):
        >>>             os.symlink(b04, outpath)
        >>>     input_dirs.append(input_dir)
        >>> #
        >>> baseline = find_baseline_scene([os.path.join(d, 'MTD_TL.xml') for d in input_dirs])
        >>> #
        >>> mgrs_tile = ''.join(safedirs[0].split(os.path.sep)[-4:-1])
        >>> # not essential, could change with demodata
        >>> assert mgrs_tile == '52SDG'
        >>> # the tile matches
        >>> assert list(baseline.keys()) == [mgrs_tile]
        >>> #
        >>> df = baseline[mgrs_tile]
        >>> assert df.shape == (1,8)
        >>> # since it only has 1 row, let it act like a dict
        >>> df = df.squeeze()
        >>> # the tile matches
        >>> df['mgrs_tile_id'] == mgrs_tile
        >>> # the granuledir exists in the chosen safedir
        >>> assert input_dirs[1] in df['granuledir']
        >>> #
        >>> # alternate use:
        >>> baseline = find_baseline_scene([os.path.join(d, 'MTD_TL.xml') for d in input_dirs], return_paths=True)
        >>> assert baseline[mgrs_tile].startswith(os.path.abspath(input_dirs[1]))
        >>> #
        >>> df.pop('granuledir')  # not portable for testing
        >>> df_dict = df.to_dict()
        >>> print(df_dict)
        >>> # Did the data on the server change? Should we ever expect that this is different?
        >>> assert (df.to_dict() == {
        >>>     'granule_id': 'S2A_MSI_L2A_T52SDG_20181104T021841_20181104T021841',
        >>>     'proc_ver': 2.06,
        >>>     'sun_zenith_angle': 53.7076919780578,
        >>>     'cloud': 0.00046,
        >>>     'coverage': 0.9257082093290998,
        >>>     'mgrs_tile_id': '52SDG',
        >>>     'score': 0.8309162895912932
        >>> }) or df.to_dict() == {
        >>>     'granule_id': 'S2A_MSI_L2A_T52SDG_20181104T021841_20181104T021841',
        >>>     'proc_ver': 2.06,
        >>>     'sun_zenith_angle': 53.7076919780578,
        >>>     'cloud': 0.00046,
        >>>     'coverage': 0.9220606683454932,
        >>>     'mgrs_tile_id': '52SDG',
        >>>     'score': 0.8300044043453915
        >>> }

    '''
    df_rows = []

    for pfname_xml in xmls:
        tmp_dict = {}

        # Granule ID
        path_to_xml = os.path.dirname(pfname_xml)
        granule_id = os.path.basename(path_to_xml)
        # Checking granule id
        if len(granule_id) == 0:
            continue
        tmp_dict['granule_id'] = granule_id
        tmp_dict['granuledir'] = path_to_xml

        # MGRS tile id from granule id without T
        tmp_dict['mgrs_tile_id'] = granule_id.split('_')[3][1:]

        # Reading XML file
        tree = ET.parse(pfname_xml)
        root = tree.getroot()

        # Getting cloud cover 0-1
        cloud = 1.  # default value
        for x in root.iter("CLOUDY_PIXEL_PERCENTAGE"):
            cloud = float(x.text) / 100.
            break
        tmp_dict['cloud'] = cloud

        # Mean sun zenith angle 0-90
        sun_zenith_angle = 90.  # default value
        for x in root.iter('Mean_Sun_Angle'):
            for c in x:
                if (c.tag == 'ZENITH_ANGLE'):
                    sun_zenith_angle = float(c.text)
        tmp_dict['sun_zenith_angle'] = sun_zenith_angle

        # Baseline processing
        proc_ver = '02.01'  # default value
        for x in root.iter("TILE_ID"):
            tile_id = x.text
            break
        proc_ver = tile_id.split('_')[-1]
        tmp_dict['proc_ver'] = float(proc_ver[1:])  # assuming 'N02.01

        # Coverage of the scene 0-1
        # Try to grab coverage value from STAC item if exists
        coverage = None
        for json_path in glob.glob(os.path.join(path_to_xml, '*.json')):
            try:
                stac_item = pystac.Item.from_file(json_path)
            except pystac.errors.STACTypeError:
                continue
            else:
                # Fall back to checking 'data_coverage' field
                data_coverage = stac_item.properties.get(
                    'sentinel:data_coverage',
                    stac_item.properties.get('data_coverage'))

                if data_coverage is not None:
                    # Scaling coverage to be 0-1
                    coverage = float(data_coverage) / 100.0
                break

        if coverage is None:
            # Compute from B04 band if couldn't grab from STAC
            pfname_b04 = glob.glob(
                os.path.join(path_to_xml, '*_B04.tif'))
            if len(pfname_b04) == 1:
                ds = gdal.Open(pfname_b04[0])
                if not (ds is None):  # checking if file is valid
                    arr = ds.GetRasterBand(1).ReadAsArray()
                    coverage = np.sum(arr > 0) / (1. * arr.shape[0] * arr.shape[1])

                    ds = None
                    arr = None

        if coverage is None:
            print("Couldn't determine coverage for '{}', setting to 0.".format(
                granule_id))
            coverage = 0.

        tmp_dict['coverage'] = coverage

        # Adding to the dataframe
        # df = df.append(tmp_dict, ignore_index=True)
        print(granule_id, proc_ver, sun_zenith_angle, cloud, coverage,
              tmp_dict['mgrs_tile_id'])

        df_rows.append(tmp_dict)

        tree = None
        # break

    df = pd.DataFrame(df_rows)
    norm_value_proc_ver = df['proc_ver'].max()
    # print(norm_value_proc_ver)
    df['score'] = 0.25 * df['coverage'] + 0.25 * (1 - df['cloud']) + \
                 0.25 * (1 - df['sun_zenith_angle'] / 90.) + 0.25 * ( df['proc_ver'] / norm_value_proc_ver)

    print(df)
    mgrs_tile_list = pd.unique(df['mgrs_tile_id'].values)
    result = {}

    for mgrs_tile in mgrs_tile_list:
        df_tmp = df[df['mgrs_tile_id'] == mgrs_tile]
        df_best = df_tmp.loc[[df_tmp['score'].astype(float).idxmax()]]
        if return_paths:
            result[mgrs_tile] = os.path.normpath(df_best.squeeze()['granuledir'])
        else:
            result[mgrs_tile] = df_best

    return result


def main():
    '''
    Look through a folder for S2 scenes, pick a baseline scene for each tile, and write them to the folder as CSVs.
    '''
    if len(sys.argv) < 2:
        print_usage()
        return

    input_folder = sys.argv[1]

    xmls = [
        safedir_to_xml(s)
        for s in glob.glob(os.path.join(input_folder, '*.SAFE'))
    ]
    for mgrs_tile, df in find_baseline_scene(xmls):
        df.to_csv(os.path.join(input_folder,
                               f'{mgrs_tile}.baseline.scene.csv'),
                  index=False)


if __name__ == '__main__':
    main()