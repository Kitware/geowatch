# UMD Coregistration
Registers S2 imagery to a baseline scene
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
stac_catalog|Input STAC catalog of images|Yes
output_dir|Directory where output images / data will be written.  Will be created if it doesn't already exist.|Yes

## Outputs:
Name|Description
---|---
stac_catalog|STAC Catalog JSON string
output_dir|Directory where images / data have been written.
