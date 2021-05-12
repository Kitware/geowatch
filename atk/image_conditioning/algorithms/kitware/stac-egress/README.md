# STAC Egress
Upload STAC catalog and assets to S3
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
stac-catalog|Input STAC catalog to be uploaded.|Yes
s3-bucket|Destination S3 bucket for data upload|Yes
dry-run|Don't actually upload to S3|Yes

## Outputs:
Name|Description
---|---
s3-bucket|S3 bucket where data has been uploaded
stac-catalog|STAC Catalog JSON string
