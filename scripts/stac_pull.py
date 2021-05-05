from pystac_client import Client
import os
import subprocess
import json
import mgrs
import argparse
import sys

def file_to_date(file):
    year = '20' + file[5:7]
    month_dict = {
        'JAN': '01',
        'FEB': '02', 
        'MAR': '03', 
        'APR': '04',
        'MAY': '05', 
        'JUN': '06', 
        'JUL': '07',
        'AUG': '08', 
        'SEP': '09', 
        'OCT': '10', 
        'NOV': '11', 
        'DEC': '12'
    }
    month = month_dict[file[2:5]]
    day = file[0:2]
    return os.path.join(year, month, day)

def pull(out_dir, AOI, datetime):
    #args:  out_dir = path/to/output/image/directory  bbox = [xmin, ymin, xmax, ymax]  datetime = ['year_min', 'year_max']
    catalog = Client.open('https://api.smart-stac.com/', headers={"x-api-key": os.environ['STAC_API_KEY']})
    catalog.to_dict()
    children = catalog.get_child_links()
    m = mgrs.MGRS()
    zone = m.toMGRS(AOI[3], AOI[0])
    for child in children:
        print(child)
    search = catalog.search(collections=['worldview-nitf'], bbox=AOI, datetime = datetime)
    print(f"{search.matched()} Items found")
    for item in search.items():
        print("Image: " + item.id)
        id = item.id
        s3 = os.path.join('s3://smart-imagery/worldview-nitf', zone[0:2], zone[2], zone[3:5], file_to_date(id), id[16:])
        print("Downloading from " + s3)
        cmd_args = ['aws', 's3', '--profile', 'iarpa', 'cp', os.path.join(s3, id+'.NTF'), os.path.join(out_dir, id+'.NTF')]
        subprocess.run(cmd_args, check=True)

    print('DONE')

def main(args):
    parser = argparse.ArgumentParser(
        description="Pull WorldView imagery from STAC")
    parser.add_argument("--AOI",
                        help="WATCH site to pull images for")
    parser.add_argument("--out_dir", help="Output directory")
    args = parser.parse_args(args)
    aoi_dict = {
        "KR-Pyeongchang": [128.662489, 37.659517, 128.676673, 37.664560],
        "AE-Dubai": [52.217275, 23.957049, 52.249111, 23.978136],
        "US-Waynesboro": [-81.776670, 33.132338, -81.764686, 33.146012],
        "BR-Rio-0277": [-43.342788, -22.960878, -43.336003, -22.954186],
        "BR-Rio-0270": [-43.438075, -22.999946, -43.432115, -23.003319]
    }
    datetime = ['2014', '2018']

    pull(args.out_dir, aoi_dict[args.AOI], datetime)

if __name__ == '__main__':
    main(sys.argv[1:])
    
    #Usage: STAC_API_KEY=<api key> python3 stac_pull.py --AOI <WATCH site> --out_dir <output directory>
    #This assumes that the user has properly configured AWS keys to access s3
