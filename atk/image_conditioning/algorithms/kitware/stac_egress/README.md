# STAC Egress
Upload STAC catalog and assets to S3
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
stac_catalog|Input STAC catalog to be uploaded.|Yes
s3_bucket|Destination S3 bucket for data upload|Yes
dry_run|Don't actually upload to S3|Yes
output_dir|If provided, the output STAC Catalog with hrefs updated based on the destination s3_bucket will also be written out to this directory.|

## Outputs:
Name|Description
---|---
s3_bucket|S3 bucket where data has been uploaded
stac_catalog|STAC Catalog JSON string
