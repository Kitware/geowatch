import json
import datetime as dt
import logging
from shapely import wkt
import requests
from PIL import Image


def from_json(labeled_data, coco_output):
    # read labelbox JSON output
    with open(labeled_data, 'r') as f:
        label_data = json.load(f)

    # setup COCO dataset container and info
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }
    print(label_data)
    print(len(label_data))
    coco['info'] = {
        'year': dt.datetime.now(dt.timezone.utc).year,
        'version': None,
        'description': label_data[0]['Project Name'],
        'contributor': label_data[0]['Created By'],
        'url': 'labelbox.com',
        'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
    }

    for data in label_data:
        print(data)
        # Download and get image name
        try:
            response = requests.get(data['Labeled Data'], stream=True)
        except requests.exceptions.MissingSchema:
            logging.exception(('"Labeled Data" field must be a URL. '
                               'Support for local files coming soon'))
            continue
        except requests.exceptions.ConnectionError:
            logging.exception('Failed to fetch image from {}'
                              .format(data['Labeled Data']))
            continue

        response.raw.decode_content = True
        im = Image.open(response.raw)
        width, height = im.size

        image = {
            "id": data['ID'],
            "width": width,
            "height": height,
            "file_name": data['Labeled Data'],
            "license": None,
            "flickr_url": data['Labeled Data'],
            "coco_url": data['Labeled Data'],
            "date_captured": None,
        }

        coco['images'].append(image)

        # convert WKT multipolygon to COCO Polygon format
        for cat in data['Label'].keys():

            try:
                # check if label category exists in 'categories' field
                cat_id = [c['id'] for c in coco['categories']
                          if c['supercategory'] == cat][0]
            except IndexError:
                cat_id = len(coco['categories']) + 1
                category = {
                    'supercategory': cat,
                    'id': len(coco['categories']) + 1,
                    'name': cat
                }
                coco['categories'].append(category)
            print(data['Label'])
            print(data['Label'][cat])
            print(type(data['Label'][cat][0]))
            print(cat)
            multipolygon = wkt.loads(data['Label'][cat])
            for m in multipolygon:
                segmentation = []
                for x, y in m.exterior.coords:
                    segmentation.extend([x, height - y])

                annotation = {
                    "id": len(coco['annotations']) + 1,
                    "image_id": data['ID'],
                    "category_id": cat_id,
                    "segmentation": [segmentation],
                    "area": m.area,  # float
                    "bbox": [m.bounds[0], m.bounds[1],
                             m.bounds[2] - m.bounds[0],
                             m.bounds[3] - m.bounds[1]],
                    "iscrowd": 0
                }

                coco['annotations'].append(annotation)

    with open(coco_output, 'w+') as f:
        f.write(json.dumps(coco))


if __name__ == "__main__":
    labeled_data = '/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/labelbox_weak_labels.json'

    # set coco_output to the file name you want the COCO data to be written to
    coco_output = '/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/fromlabelbox.kwcoco.json'

    # call the Labelbox to COCO conversion
    from_json(labeled_data=labeled_data, coco_output=coco_output)
