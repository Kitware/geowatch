from pystac_client import Client
import os
import subprocess
import json
import argparse
import sys

def pull(out_dir, AOI, daterange, api_key, endpoint):  
    '''args:  out_dir = path/to/output/image/directory  
              AOI = [xmin, ymin, xmax, ymax]  
              datetime = ['year_min', 'year_max']

       returns: retrieved catalog as a dict, ready to be dumped as a JSON
    '''
    catalog = Client.open(endpoint, headers={"x-api-key": api_key})
    children = catalog.get_child_links()
    for child in children:
        print(child)
    search = catalog.search(collections=['worldview-nitf'], 
                            bbox=AOI, datetime=daterange)
    print(f"{search.matched()} Items found")
    for item in search.items():
        s3 = item.assets['data'].get_absolute_href()
        cmd_args = ['aws', 's3', 
                    '--profile', 'iarpa', 
                    'cp', s3, 
                    os.path.join(out_dir, item.id+'.NTF')]
        subprocess.run(cmd_args, check=True)
    return catalog.to_dict()

def main(args):
    parser = argparse.ArgumentParser(
        description="Pull WorldView imagery from STAC. This assumes that the \
                     user has properly configured AWS keys to access s3.")
    parser.add_argument("--AOI", help="Site to pull images for. If a bbox is \
                        provided to define a new site, this will be used as \
                        its name")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--endpoint", help="STAC API endpoint")
    parser.add_argument("--api_key", help="STAC API key", 
                        default=os.environ['STAC_API_KEY'])
    parser.add_argument("--bbox", help="User-defined AOI as a JSON string: \
                                        '[<xmin>, <ymin>, <xmax>, <ymax>]'", 
                        default=None)
    parser.add_argument("--daterange", 
                        help="JSON string giving date range to pull images for:\
                        '[\"<start date>\", \"<end date>\"]'. \
                        Dates should be in UTC format.", 
                        default=None)
    parser.add_argument("--jsondump", 
                        help="Dump the retrieved STAC catalog as a JSON", 
                        action="store_true")
    args = parser.parse_args(args)
    aoi_dict = {
        "KR-Pyeongchang": [128.662489, 37.659517, 128.676673, 37.664560],
        "AE-Dubai": [52.217275, 23.957049, 52.249111, 23.978136],
        "US-Waynesboro": [-81.776670, 33.132338, -81.764686, 33.146012],
        "BR-Rio-0277": [-43.342788, -22.960878, -43.336003, -22.954186],
        "BR-Rio-0270": [-43.438075, -22.999946, -43.432115, -23.003319]
    }
    date_dict = {
        "KR-Pyeongchang": ['2014', '2018'],
        "AE-Dubai": ['2013', '2020'],
        "US-Waynesboro": ['2013', '2019'],
        "BR-Rio-0277": ['2014', '2019'],
        "BR-Rio-0270": ['2014', '2019']
    }
    if args.bbox:
        aoi_dict[args.AOI] = json.loads(args.bbox)
    if args.daterange:
        date_dict[args.AOI] = json.loads(args.daterange)
    search = pull(args.out_dir, 
                  aoi_dict[args.AOI], 
                  date_dict[args.AOI], 
                  args.api_key, 
                  args.endpoint)
    if args.jsondump:
        with open(os.path.join(args.out_dir, 
                  args.AOI + '_catalog.json'), 'w') as f:
            json.dump(search, f)

if __name__ == '__main__':
    main(sys.argv[1:])
