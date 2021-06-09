import os
import json
import subprocess
import tempfile
import copy

from algorithm_toolkit import Algorithm, AlgorithmChain


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        stac_catalog = copy.deepcopy(params['stac_catalog'])

        # Upload each feature's assets from the STAC catalog
        for feature in stac_catalog.get('features', ()):
            feature_id = feature['id']
            local_path = feature['assets']['data']['href']

            local_basename = os.path.basename(local_path)

            upload_path = os.path.join(
                params['s3_bucket'], feature_id, local_basename)

            command = ['aws', 's3', '--profile', 'iarpa', 'cp']
            if params['dry_run'] == 1:
                command.append('--dryrun')

            command.extend([local_path, upload_path])

            # TODO: Manually check return code / output
            self.logger.info("Running: {}".format(' '.join(command)))
            if params['dry_run'] != 1:
                subprocess.run(command, check=True)

            # Update feature asset href to point to uploaded path on
            # S3
            feature['assets']['data']['href'] = upload_path

        # Upload catalog as well
        with tempfile.NamedTemporaryFile('w', suffix='.json') as out_catalog:
            out_catalog.write(json.dumps(stac_catalog, indent=2))
            out_catalog.flush()

            catalog_upload_path = os.path.join(
                params['s3_bucket'], 'catalog.json')

            command = ['aws', 's3', '--profile', 'iarpa', 'cp']
            if params['dry_run'] == 1:
                command.append('--dryrun')

            command.extend([out_catalog.name, catalog_upload_path])

            # TODO: Manually check return code / output
            self.logger.info("Running: {}".format(' '.join(command)))
            subprocess.run(command, check=True)

        cl.add_to_metadata('stac_catalog', stac_catalog)
        cl.add_to_metadata('s3_bucket', params['s3_bucket'])

        # Do not edit below this line
        return cl
