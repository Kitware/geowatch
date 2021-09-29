# Baseline Framework Ingress
Ingress data from T&E Baseline Framework input JSON file.
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
input_path|Path to input JSON file|Yes
output_dir|Output directory for downloaded assets|Yes
aws_profile|Optionally provide an AWS profile for AWS CLI commands|
dryrun|Don't actually download remote assets|
requester_pays|Confirm that some S3 assets may require requester to pay|

## Outputs:
Name|Description
---|---
stac_catalog|STAC Catalog JSON string
