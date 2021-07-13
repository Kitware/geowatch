import json
import os
import random
import argparse
import sys
import re

import requests


http_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
all_chars = http_chars + '`~!@#$%*()-_=+[]{}|;:,./?'
ENV_VAR_RE = re.compile(r'^\$(\w+)$')


def create_random_string(n=20, http_safe=False):
    if http_safe is False:
        chars = all_chars
    else:
        chars = http_chars
    temp_str = "".join([random.SystemRandom().choice(chars) for i in range(n)])
    return temp_str


# how to run a chain with requests
def run_atk_chain(api_key, chain_json, node_address, status_key=None):
    if status_key is None:
        status_key = create_random_string(http_safe=True)
    request_json = {
        "api_key": api_key,
        "chain": json.dumps(chain_json),
        "output_type": "stdout",
        "status_key": status_key
    }

    print(node_address)
    r = requests.post(node_address, data=request_json, verify=False)
    try:
        print(r.text)
        res = json.loads(r.text)
    except json.decoder.JSONDecodeError:
        res = json.loads('["Error calling algorithm node."]')

    return status_key, res


def list_env_replace(list_):
    out_list = []
    for item in list_:
        if isinstance(item, dict):
            out_list.append(dict_env_replace(item))
        elif isinstance(item, list):
            out_list.append(list_env_replace(item))
        elif isinstance(item, str):
            m = re.match(ENV_VAR_RE, item)
            if m is not None:
                out_list.append(os.environ.get(m.group(1), item))
            else:
                out_list.append(item)
        else:
            out_list.append(item)

    return out_list


def dict_env_replace(dict_):
    out_dict = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            out_dict[key] = dict_env_replace(value)
        elif isinstance(value, list):
            out_dict[key] = list_env_replace(value)
        elif isinstance(value, str):
            m = re.match(ENV_VAR_RE, value)
            if m is not None:
                out_dict[key] = os.environ.get(m.group(1), value)
            else:
                out_dict[key] = value
        else:
            out_dict[key] = value

    return out_dict


# example_chain = {
#     "chain_name": "ingest_and_align",
#     "algorithms": [
#         {
#             "name": "kitware/stac_ingress",
#             "parameters": {
#                 "stac_api_key": os.environ["STAC_API_KEY"],
#                 "aoi_bounds": [128.662489, 37.659517, 128.676673, 37.664560],
#                 "date_range": ["2017", "2018"],
#                 "stac_api_url": "https://api.smart-stac.com/",
#                 "output_dir": "./tmp/ingress",
#                 "collections": ["worldview-nitf"],
#                 "dry_run": 1,
#                 "max_results": 2
#             }
#         },
#         {
#             "name": "kitware/align_crs",
#             "parameters": {
#                 "stac_catalog": {
#                     "source": "chain_ledger",
#                     "source_algorithm": "kitware/stac_ingress",
#                     "key": "stac_catalog",
#                     "occurrence": "first"
#                 },
#                 "aoi_bounds": [128.662489, 37.659517, 128.676673, 37.664560],
#                 "output_dir": "./tmp/aligned"
#             }
#         }
#     ]
# }


def main():
    parser = argparse.ArgumentParser(
        description="Execute chain against running ATK web service")

    parser.add_argument('chain_json',
                        type=str,
                        help="Path to filled out chain JSON file")
    parser.add_argument("--url",
                        type=str,
                        required=True,
                        help="URL of running ATK web service")
    parser.add_argument("--api_key",
                        type=str,
                        help="ATK web service API key")

    run_chain(**vars(parser.parse_args()))


def run_chain(chain_json, url, api_key=None):
    if api_key is None:
        api_key = os.environ['ATK_API_KEY']

    with open(chain_json, 'r') as f:
        chain = json.load(f)

    # Replace environment variables specified in chain JSON
    chain = dict_env_replace(chain)

    chain_path = 'chains/{}/'.format(chain['chain_name'])

    if not url.endswith(chain_path):
        url = "{}/{}".format(url, chain_path)

    status, response = run_atk_chain(
        api_key, chain, url)

    print(status)
    print(response)

    return response


if __name__ == "__main__":
    sys.exit(main())
