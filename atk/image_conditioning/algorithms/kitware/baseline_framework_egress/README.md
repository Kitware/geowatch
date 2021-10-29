# Baseline Framework Egress
Convert STAC Catalog to T&E baseline framework format and upload to S3
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
stac_catalog|Input STAC Catalog to convert and egress|Yes
output_path|S3 output path for converted STAC Catalog|Yes
output_bucket|Output S3 Bucket where output STAC Items and their assets will be written|Yes
aws_profile|Optionally provide an AWS profile for AWS CLI commands|
dryrun|Don't actually upload anything to S3|

## Outputs:
Name|Description
---|---
output_path|S3 output path where converted STAC Catalog JSON was written
output_bucket|Output S3 Bucket where output STAC Items and their assets were written
te_output|T&E baseline framework formatted output
