import tempfile
import subprocess
import json
import os

import pystac

from watch.cli.baseline_framework_ingress import ingress_item
from watch.cli.baseline_framework_egress import egress_item


class CacheItemOutputS3Wrapper:
    def __init__(self, item_map, outbucket, aws_profile=None):
        self.item_map = item_map
        self.outbucket = outbucket

        if aws_profile is not None:
            self.aws_base_command =\
              ['aws', 's3', '--profile', aws_profile, 'cp', '--no-progress']
        else:
            self.aws_base_command = ['aws', 's3', 'cp', '--no-progress']

    def __call__(self, stac_item, *args, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            status_file_basename = '{}.done'.format(stac_item['id'])
            status_item_s3_path = os.path.join(
                self.outbucket, 'status', status_file_basename)
            status_item_local_path = os.path.join(
                tmpdirname, status_file_basename)

            try:
                subprocess.run([*self.aws_base_command,
                                status_item_s3_path,
                                status_item_local_path],
                               check=True)
            except subprocess.CalledProcessError:
                pass
            else:
                print("* Item: {} previously processed, not "
                      "re-processing".format(stac_item['id']))
                with open(status_item_local_path) as f:
                    return [json.loads(line) for line in f]

            output_stac_items = self.item_map(stac_item, *args, **kwargs)

            output_status_file = os.path.join(
                tmpdirname, '{}.output.done'.format(stac_item['id']))
            with open(output_status_file, 'w') as outf:
                for output_item in output_stac_items:
                    if isinstance(output_item, pystac.Item):
                        print(json.dumps(output_item.to_dict()), file=outf)
                    else:
                        print(json.dumps(output_item), file=outf)

            subprocess.run([*self.aws_base_command,
                            output_status_file,
                            status_item_s3_path], check=True)

            return output_stac_items


def _default_item_selector(stac_item):
    return True


class IngressProcessEgressWrapper:
    def __init__(self,
                 item_map,
                 outbucket,
                 aws_base_command,
                 dryrun=False,
                 stac_item_selector=_default_item_selector):
        self.item_map = item_map
        self.outbucket = outbucket
        self.aws_base_command = aws_base_command
        self.dryrun = dryrun
        self.stac_item_selector = stac_item_selector

    def __call__(self, stac_item, *args, **kwargs):
        # Assumes that the 'self.item_map' function accepts
        # 'stac_item' and 'working_dir' arguments. TODO: actually
        # check this via introspection
        print("* Processing item: {}".format(stac_item['id']))

        if not self.stac_item_selector(stac_item):
            print("**  STAC item {} did not satisfy selector, not "
                  "processing".format(stac_item['id'])
            return [stac_item]

        with tempfile.TemporaryDirectory() as tmpdirname:
            ingressed_item = ingress_item(
                stac_item,
                os.path.join(tmpdirname, 'ingress'),
                self.aws_base_command,
                self.dryrun)

            processed_item = self.item_map(
                ingressed_item,
                tmpdirname,
                *args, **kwargs)

            processed_items = []
            if isinstance(processed_item, dict):
                processed_items.append(pystac.Item.from_dict(processed_item))
            elif isinstance(processed_item, pystac.Item):
                processed_items.append(processed_item)
            else:
                # Assume already an iterable of pystac.Item
                processed_items = processed_item

            output_items = []
            for item in processed_items:
                output_items.append(egress_item(item,
                                                self.outbucket,
                                                self.aws_base_command))

            # Returning a list here
            return output_items
