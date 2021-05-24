# Align CRS
Set all images to the same UTM CRS.
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
stac_catalog|Input STAC catalog of images|Yes
aoi_bounds|Spatial bounds of the Area of Interest (AOI).|Yes
output_dir|Directory where output images / data will be written.  Will be created if it doesn't already exist.|Yes

## Outputs:
Name|Description
---|---
stac_catalog|STAC Catalog JSON string
output_dir|Directory where images / data have been written.
