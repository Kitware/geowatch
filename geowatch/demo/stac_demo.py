import os
import ubelt as ub
from geowatch.demo import sentinel2_demodata
import json


def demo():
    """
    Returns:
        str: path to a demo stac catalog
    """
    data = sentinel2_demodata.grab_sentinel2_product(0)
    gpath = os.path.join(str(data.path), str(data.images[0]))
    dpath = ub.Path.appdir('geowatch/demo/demo_stac').ensuredir()
    cat_path = os.path.join(dpath, 'demo_catalog.json')
    expected_cat_sha512 = '78e9f307e4e365c826a55f6beaa9d6e3b6b6ba4e3cc7d08'
    expected_item1_sha512 = '78224b4b2e56cbc58ca8d76b632b1c2c323eeae653b5323'
    expected_item2_sha512 = '348daaffb70c17da43f08f5b1506f929467f6986e7e8170'
    catalog = {
        "id": "demo catalog",
        "stac_version": "1.0.0-beta.2",
        "description": "demo catalog",
        "links": [
            {
                "rel": "root",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "self",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "item",
                "href": os.path.join(dpath, "testitem1.json"),
                "type": "application/json"
            },
            {
                "rel": "item",
                "href": os.path.join(dpath, "testitem2.json"),
                "type": "application/json"
            }
        ]
    }
    if (not os.path.exists(cat_path) or
            not ub.hash_file(cat_path).startswith(expected_cat_sha512)):
        with open(cat_path, 'w') as f:
            json.dump(catalog, f)
    path1 = os.path.join(dpath, "testitem1.json")
    item1 = {
        "type": "Feature",
        "stac_version": "1.0.0-beta.2",
        "id": "testitem1",
        "properties": {
            "datetime": "2018-11-06T02:17:10.060000Z",
        },
        "geometry": {},
        "links": [
            {
                "rel": "root",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "parent",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "self",
                "href": path1,
                "type": "application/json"
            }
        ],
        "assets": {
            "data": {
                "href": gpath,
                "type": "application/vnd.nitf"
            }
        },
        "bbox": [],
        "stac_extensions": []
    }
    if (not os.path.exists(path1) or
            not ub.hash_file(path1).startswith(expected_item1_sha512)):
        with open(os.path.join(dpath, "testitem1.json"), 'w') as f:
            json.dump(item1, f)
    path2 = os.path.join(dpath, "testitem2.json")
    item2 = {
        "type": "Feature",
        "stac_version": "1.0.0-beta.2",
        "id": "testitem2",
        "properties": {
            "datetime": "2018-11-06T02:17:10.060000Z",
        },
        "geometry": {},
        "links": [
            {
                "rel": "root",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "parent",
                "href": cat_path,
                "type": "application/json"
            },
            {
                "rel": "self",
                "href": path2,
                "type": "application/json"
            }
        ],
        "assets": {
            "testimg1": {
                "href": gpath,
                "type": "application/vnd.nitf",
                "roles": ["data"]
            },
            "testimg2": {
                "href": gpath,
                "type": "application/vnd.nitf",
                "roles": ["data"]
            }
        },
        "bbox": [],
        "stac_extensions": []
    }
    if (not os.path.exists(path2) or
            not ub.hash_file(path2).startswith(expected_item2_sha512)):
        with open(os.path.join(dpath, "testitem2.json"), 'w') as f:
            json.dump(item2, f)
    return cat_path


if __name__ == '__main__':
    print(demo())
